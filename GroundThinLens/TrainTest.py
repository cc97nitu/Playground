import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps
from ThinLens.Models import F0D0Model, SIS18_Cell_minimal, SIS18_Cell, \
    SIS18_Lattice_minimal, SIS18_Lattice

import matplotlib.pyplot as plt
import time
import cpymad.madx

from SampleBeam import Beam
import tools.madX
import tools.plot
import tools.tuneFFT


def trainLoop(model, criterion, optimizer, epochs: int):
    # train loop

    for epoch in range(epochs):
        for i, data in enumerate(trainLoader, 0):
            bunch, label = data[0], data[1]

            optimizer.zero_grad()

            out = model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)

            # loss = criterion(out, label)  # full phase-space
            loss = criterion(out[:, [0, 2], :], label[:, [0, 2], :])  # only x-, y-plane

            loss.backward()
            optimizer.step()

        # if epoch % 10 == 9:
        #     print(loss.item())
        print(loss.item())

    return


def trainPerCell(model, criterion, optimizer, epochs: int):
    mapCount: int = 0

    for cell in model.cells:
        # current cell quads only
        model.requires_grad_(False)
        for element in cell:
            if type(element) is Elements.Quadrupole:
                element.requires_grad_(True)

        # count model outputs up to current cell
        if outputPerElement:
            mapCount += len(cell)
        else:
            for element in cell:
                if type(element) is Elements.Monitor:
                    mapCount += 1

        # train loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            out = model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)

            # loss = criterion(out, label)  # full phase-space
            # loss = criterion(out[:, [0, 2], :], label[:, [0, 2], :])  # only x-, y-plane
            loss = criterion(out[:, [0, 2], :mapCount + 1], label[:, [0, 2], :mapCount + 1])  # only x-, y-plane

            loss.backward()
            optimizer.step()

            if epoch % 20 == 19:
                print(loss.item())

    return


def trainPerElement(model, criterion, optimizer, epochs: int):
    if outputPerElement:
        mapCount = 0
        for element in model.elements:
            mapCount += 1
    else:
        mapCount = 0
        for element in model.elements:
            if type(element) is Elements.Monitor:
                mapCount += 1

    print("iterating over {} positions".format(mapCount))
    print("output shape {}".format(model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)
                                   .shape))
    for m in range(mapCount):
        print("position {}/{}".format(m, mapCount))

        # train loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            out = model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)

            loss = criterion(out[:, [0, 2], :m + 1], label[:, [0, 2], :m + 1])  # only x-, y-plane

            loss.backward()
            optimizer.step()

            if epoch % 10 == 9:
                print(loss.item())

    return


if __name__ == "__main__":
    # general
    # torch.set_printoptions(precision=4, sci_mode=False)

    dim = 6
    slices = 4
    quadSliceMultiplicity = 1
    dtype = torch.double
    device = torch.device("cpu")
    outputPerElement = True  # exceeds outputAtBPM
    outputAtBPM = True

    # set up models
    # model = F0D0Model(k1=0.3, slices=slices, dim=dim, dtype=dtype)
    # perturbedModel = F0D0Model(k1=0.2, dim=dim, slices=slices, dtype=dtype)

    Lattice = SIS18_Lattice

    startK1f = 0.3217252108633675
    startK1d = -0.49177734861791

    model = Lattice(k1f=startK1f, k1d=startK1d, dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity, dtype=dtype, cellsIdentical=False).to(device)


    perturbedModel = Lattice(dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                             dtype=dtype).to(device)

    model.requires_grad_(False)
    perturbedModel.requires_grad_(False)

    # dump off
    fileName = "/dev/shm/third.json"
    with open(fileName, "w") as file:
        model.dumpJSON(file)

    # train set
    if dim == 4:
        bunch = torch.tensor([[1e-3, 2e-3, 1e-3, 0], [-1e-3, 1e-3, 0, 1e-3]], dtype=dtype).to(device)
        label = perturbedModel(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM).to(device)
    elif dim == 6:
        beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.000, particles=500)
        bunch = beam.bunch[:].to(device)
        label = perturbedModel(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM).to(device)

    trainSet = torch.utils.data.TensorDataset(bunch, label)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=25,
                                              shuffle=True, num_workers=2)

    # initial tunes
    print("initial model tune from Mad-X: {}".format(tools.madX.tune(model.madX())))

    # # fft
    # if dim == 6:
    #     print("performing fft of initial models")
    #     fft = tools.tuneFFT.getTuneChromaticity(model, turns=200, dtype=dtype, beam=beam)
    #     print("ideal model", fft)
    #     fft = tools.tuneFFT.getTuneChromaticity(perturbedModel, turns=200, dtype=dtype, beam=beam)
    #     print("perturbed model", fft)

    # plot initial trajectories
    figTrajectories, axesTrajectories = plt.subplots(3, sharex=True)
    tools.plot.trajectories(axesTrajectories[0], tools.plot.track(model, bunch, 1), model)
    axesTrajectories[0].set_ylabel("ideal")

    tools.plot.trajectories(axesTrajectories[1], tools.plot.track(perturbedModel, bunch, 1), perturbedModel)
    axesTrajectories[1].set_ylabel("perturbed")

    # plot initial beta
    figBeta, axesBeta = plt.subplots(3, sharex=True)
    tools.plot.betaMadX(axesBeta[0], model.madX(), )
    axesBeta[0].set_ylabel("ideal")

    tools.plot.betaMadX(axesBeta[1], perturbedModel.madX(), )
    axesBeta[1].set_ylabel("perturbed")

    # test symplecticity
    if dim == 4:
        sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)
    else:
        sym = torch.tensor([[0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0]], dtype=dtype)

    rMatrix = model.rMatrix()
    res = torch.matmul(rMatrix.transpose(1, 0), torch.matmul(sym, rMatrix)) - sym
    print("sym penalty before training: {}".format(torch.norm(res)))

    # activate gradients on kick maps
    for element in model.elements:
        if type(element) is Elements.Quadrupole:
            element.k1n.requires_grad_(True)

            # for m in element.maps:
            #     if type(m) is Maps.DriftMap:
            #         continue
            #     m.weight.requires_grad_(True)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    # optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # optimizer = optim.Adam(model.parameters(), lr=5e-6)
    criterion = nn.MSELoss()

    # call train loop
    print("start training")
    t0 = time.time()

    trainLoop(model, criterion, optimizer, int(5e1))
    # trainPerElement(model, criterion, optimizer, int(40))
    # trainPerCell(model, criterion, optimizer, int(40))

    print("training completed within {:.2f}s".format(time.time() - t0))

    # plot final trajectories
    tools.plot.trajectories(axesTrajectories[2], tools.plot.track(model, bunch, 1), model)
    axesTrajectories[2].set_ylabel("trained")

    axesTrajectories[2].set_xlabel("pos / m")

    figTrajectories.show()

    # plot final beta
    try:
        tools.plot.betaMadX(axesBeta[2], model.madX())
    except cpymad.madx.TwissFailed:
        print("twiss failed for trained model")
        axesBeta[2].plot(model.endPositions, torch.zeros(len(model.endPositions)))

    axesBeta[2].set_ylabel("trained")
    axesBeta[2].set_xlabel("pos / m")

    figBeta.show()

    # look at quad weights
    print("weights")
    for m in model.modules():
        if type(m) is Elements.Quadrupole:
            for mod in m.modules():
                if type(mod) is Maps.QuadKick:
                    print(mod.weight)


    # test symplecticity
    rMatrix = model.rMatrix()
    res = torch.matmul(rMatrix.transpose(1, 0), torch.matmul(sym, rMatrix)) - sym
    print("sym penalty after training: {}".format(torch.norm(res)))

    print("transport matrix after training:")
    print(model.rMatrix())

    print("determinant after training: {}".format(torch.det(model.rMatrix())))
    # # look at maps
    # print("trained model:")
    # quad = model.elements[1]
    # print(quad.rMatrix())
    # # for m in quad.maps:
    # #     print(m.rMatrix())
    #
    # print("perturbed model first quad:")
    # print(perturbedModel.elements[1].rMatrix())

    # print tunes
    print("perturbed model tunes: {}".format(perturbedModel.getTunes()))
    print("trained model tunes: {}".format(model.getTunes()))
    print("perturbed model tunes from Mad-X: {}".format(tools.madX.tune(perturbedModel.madX())))
    print("trained model tunes from Mad-X: {}".format(tools.madX.tune(model.madX())))

    # # fft
    # if dim == 6:
    #     print("performing fft of trained model")
    #     fft = tools.tuneFFT.getTuneChromaticity(model, turns=200, dtype=dtype, beam=beam)
    #     print(fft)

    # dump off
    with open(fileName, "w") as file:
        model.dumpJSON(file)
