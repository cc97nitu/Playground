import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps
from ThinLens.Beam import Beam

from ThinLens.Models import F0D0Model, SIS18_Cell_minimal, SIS18_Cell, \
    SIS18_Lattice_minimal, SIS18_Lattice

import matplotlib.pyplot as plt
import time
import cpymad.madx

import tools.madX
import tools.plot
import tools.tuneFFT

# general
# torch.set_printoptions(precision=4, sci_mode=False)

dim = 6
slices = 4
quadSliceMultiplicity = 1
dtype = torch.double
device = torch.device("cpu")
outputPerElement = False  # exceeds outputAtBPM
outputAtBPM = True

# set up models
Lattice = SIS18_Lattice

model = Lattice(dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                dtype=dtype, cellsIdentical=False).to(device)

perturbedModel = Lattice(dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                         dtype=dtype).to(device)

model.requires_grad_(False)
perturbedModel.requires_grad_(False)

# perturb model
std = 1e-2
for element in perturbedModel.elements:
    if type(element) is Elements.Quadrupole:
        k1n = element.k1n.item()
        k1n = k1n + torch.normal(0, std, size=(1,))
        element.k1n = nn.Parameter(k1n)
        element.shareWeights()

for i in range(len(model.elements)):
    if type(model.elements[i]) is Elements.Quadrupole:
        print(model.elements[i].k1n.item() - perturbedModel.elements[i].k1n.item())


# train set
beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.000, particles=500)
bunch = beam.bunch[:].to(device)
label = perturbedModel(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM).to(device)

trainSet = torch.utils.data.TensorDataset(bunch, label)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=25,
                                          shuffle=True, num_workers=2)

# plot initial trajectories
figTrajectories, axesTrajectories = plt.subplots(3, sharex=True)
tools.plot.trajectories(axesTrajectories[0], tools.plot.track(model, bunch, 1), model)
axesTrajectories[0].set_ylabel("ideal")

tools.plot.trajectories(axesTrajectories[1], tools.plot.track(perturbedModel, bunch, 1), perturbedModel)
axesTrajectories[1].set_ylabel("perturbed")

# plot inital beta
figBeta, axesBeta = plt.subplots(3, sharex=True)
tools.plot.betaMadX(axesBeta[0], model.madX(), )
axesBeta[0].set_ylabel("ideal")

tools.plot.betaMadX(axesBeta[1], perturbedModel.madX(), )
axesBeta[1].set_ylabel("perturbed")

plt.show()
