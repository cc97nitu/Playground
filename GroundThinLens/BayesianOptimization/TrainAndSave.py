#!/usr/bin/env python
# coding: utf-8

import os.path
import time
import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from cpymad.madx import Madx

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps
from ThinLens.Models import SIS18_Lattice

from GroundThinLens.SampleBeam import Beam

# arguments from CLI
q1 = float(sys.argv[1])
q2 = float(sys.argv[2])
step = float(sys.argv[3])

# general settings
dumpRequired = False

dim = 6
slices = 4
quadSliceMultiplicity = 2
dtype = torch.double
device = torch.device("cpu")
outputPerElement = False  # exceeds outputAtBPM
outputAtBPM = True

# obtain quad settings from Mad-X
output = True
madx = Madx(stdout=False)
madx.options.echo = output
madx.options.warn = output
madx.options.info = output

# specify beam
assert madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=19/18.798)

# activate sequence
sequence = "SIS18"
madx.call("Lattice/SIS18.seq")
madx.input("""
iqf=3.12391e-01;
iqd=-4.78047e-01;

ik2f=0.0;
ik2d=-0.0;
""")
madx.command.use(sequence=sequence)

# match tune
madx.input("""
match, sequence={};
global, sequence={}, q1={}, q2={};
vary, name=iqf, step=0.00001;
vary, name=iqd, step=0.00001;
lmdif, calls=500, tolerance=1.0e-10;
endmatch;
""".format(sequence, sequence, q1, q2))

iqf, iqd = madx.globals["iqf"], madx.globals["iqd"]

# prepare models
Lattice = SIS18_Lattice

model = Lattice(k1f=iqf, k1d=iqd, dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                dtype=dtype, cellsIdentical=False).to(device)

# workingpoint: q1=4.2, q2=3.3
perturbedModel = Lattice(dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                         dtype=dtype).to(device)

model.requires_grad_(False)
perturbedModel.requires_grad_(False)

if dumpRequired:
    # dump off
    fileName = "dump/{}_q1={},q2={}.json".format(type(model).__name__, iqf, iqd)

    if not os.path.isfile(fileName):
        fPath = os.path.abspath(fileName)

        with open(fileName, "w") as file:
            model.dumpJSON(file)
    else:
        raise IOError()

# create training data
beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.00, sige=0.000, particles=500)
bunch = beam.bunch[:].to(device)
label = perturbedModel(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM).to(device)

trainSet = torch.utils.data.TensorDataset(bunch, label)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=25,
                                          shuffle=True, num_workers=2)

# activate gradients on kick maps
for m in model.modules():
    if type(m) is Elements.Quadrupole:
        for mod in m.modules():
            if type(mod) is Maps.QuadKick:
                mod.requires_grad_(True)

# set up optimizer
optimizer = optim.Adam(model.parameters(), lr=step)

# loss function
criterion = nn.MSELoss()


# train model
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

    return loss.item()

# train
t0 = time.time()

finalLoss = trainLoop(model, criterion, optimizer, int(2e0))

print("training completed within {:.2f}s".format(time.time() - t0))

print("final loss: {}".format(finalLoss))

if dumpRequired:
    # save results
    with open(fileName, "w") as file:
        model.dumpJSON(file)

