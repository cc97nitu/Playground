
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import time
import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps
from ThinLens.Models import F0D0Model, SIS18_Cell_minimal, SIS18_Cell,     SIS18_Lattice_minimal, SIS18_Lattice

from GroundThinLens.SampleBeam import Beam


# general settings
dim = 6
slices = 4
quadSliceMultiplicity = 2
dtype = torch.double
device = torch.device("cpu")
outputPerElement = False  # exceeds outputAtBPM
outputAtBPM = True

# prepare models
Lattice = SIS18_Lattice

startK1f = float(sys.argv[1])
startK1d = float(sys.argv[2])
model = Lattice(k1f=startK1f, k1d=startK1d, dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                dtype=dtype, cellsIdentical=False).to(device)

# workingpoint: q1=4.2, q2=3.3
perturbedModel = Lattice(dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                         dtype=dtype).to(device)

model.requires_grad_(False)
perturbedModel.requires_grad_(False)

# dump off
fileName = "trainDump/{}_q1={},q2={}.json".format(type(model).__name__, startK1f, startK1d)

if not os.path.isfile(fileName):
    fPath = os.path.abspath(fileName)
    print(fPath)
    
    with open(fileName, "w") as file:
        model.dumpJSON(file)
else:
    raise IOError()



# create training data

# train set
if dim == 6:
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
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
# optimizer = optim.Adam(model.parameters(), lr=5e-6)

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

        print(loss.item())

    return

# train
t0 = time.time()

trainLoop(model, criterion, optimizer, int(2e1))

print("training completed within {:.2f}s".format(time.time() - t0))


# save results
with open(fileName, "w") as file:
    model.dumpJSON(file)

