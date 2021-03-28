import sys
import os
import time

import numpy as np

import torch.nn as nn
import torch.optim
import torch.utils.data

from cpymad.madx import Madx

from TorchOcelot.Lattice import SIS18_Lattice
from TorchOcelot.Models import LinearModel


# general settings
dim = 6
Lattice = SIS18_Lattice

outputPerElement = False
outputAtBPM = True

dtype = torch.double
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("running on {}".format(device))

# arguments from CLI
dumpPath = sys.argv[1]
if dumpPath == "False":
    dumpRequired = False
else:
    dumpRequired = True

q1 = float(sys.argv[2])
q2 = float(sys.argv[3])
step = float(sys.argv[4])

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
madx.call("/home/dylan/ThesisWorkspace/BayesianOptimization/Lattice/SIS18.seq")
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

# create model of SIS18
model = LinearModel(Lattice(k1f=iqf, k1d=iqd), dim, dtype)
model = model.to(device)
model.setTrainable("quadrupoles")

# workingpoint: q1=4.2, q2=3.3
idealModel = LinearModel(Lattice(), dim, dtype)
idealModel = idealModel.to(device)
idealModel.requires_grad_(False)

if dumpRequired:
    # dump off
    fileName = "{}_q1={},q2={},step={}.json".format(type(model).__name__, q1, q2, step)
    fileName = os.path.join(dumpPath, fileName)

    if not os.path.isfile(fileName):
        fPath = os.path.abspath(fileName)

        with open(fileName, "w") as file:
            model.dumpJSON(file)
    else:
        raise FileExistsError(fileName)

# load bunch
bunch = np.loadtxt("../../../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.as_tensor(bunch, dtype=dtype)[:20,:dim]
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim
bunch = bunch.to(device)

# build training set from ideal model
with torch.no_grad():
    bunchLabels = idealModel(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)

bunch = bunch.to("cpu")
bunchLabels = bunchLabels.to("cpu")

trainSet = torch.utils.data.TensorDataset(bunch, bunchLabels)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=100,
                                          shuffle=True, num_workers=2)

# set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=step)
criterion = nn.MSELoss()


# train model
def trainLoop(model, criterion, optimizer, epochs: int):
    # train loop

    for epoch in range(epochs):
        for i, data in enumerate(trainLoader, 0):
            inputs, label = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            out = model(inputs, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)

            # loss = criterion(out, label) + model.symplecticRegularization() # full phase-space
            loss = criterion(out[:, [0, 2], :], label[:, [0, 2], :]) + model.symplecticRegularization()  # only x-, y-plane

            loss.backward()

            # perform step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

    return loss.item()

# train
t0 = time.time()

finalLoss = trainLoop(model, criterion, optimizer, int(2.5e3))

print("training completed within {:.2f}s".format(time.time() - t0))

print("final loss: {}".format(criterion(model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM), bunchLabels),))

if dumpRequired:
    # save results
    with open(fileName, "w") as file:
        model.dumpJSON(file)
