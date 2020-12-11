import torch
import torch.nn as nn
import torch.optim as optim

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps
from ThinLens.Models import F0D0Model, SIS18_Cell_minimal, SIS18_DoubleCell_minimal, SIS18_DoubleCell_minimal_identical, \
    SIS18_Lattice_minimal

import matplotlib.pyplot as plt

import tools.madX
import tools.plot

# general
torch.set_printoptions(precision=4, sci_mode=False)

dim = 4
slices = 10
quadSliceMultiplicity = 1
dtype = torch.double
outputPerElement = True

# set up models
# model = F0D0Model(k1=0.3, slices=slices, dim=dim, dtype=dtype)
# perturbedModel = F0D0Model(k1=0.2, dim=dim, slices=slices, dtype=dtype)

Lattice = SIS18_Lattice_minimal
model = Lattice(dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity, dtype=dtype)

k1f = 3.39177e-01
k1d = -5.26301e-01
perturbedModel = Lattice(k1f=k1f, k1d=k1d, dim=dim, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity,
                         dtype=dtype)

model.requires_grad_(False)
perturbedModel.requires_grad_(False)

# train set
bunch = torch.tensor([[1e-3, 2e-3, 1e-3, 0], [-1e-3, 1e-3, 0, 1e-3]], dtype=dtype)
label = perturbedModel(bunch, outputPerElement=outputPerElement)

# initial tunes
print("initial model tune from Mad-X: {}".format(tools.madX.tune(model.madX())))

# plot initial trajectories
fig, axes = plt.subplots(3, sharex=True)
tools.plot.trajectories(axes[0], tools.plot.track(model, bunch, 1), model)
axes[0].set_ylabel("ideal")

tools.plot.trajectories(axes[1], tools.plot.track(perturbedModel, bunch, 1), perturbedModel)
axes[1].set_ylabel("perturbed")

# test symplecticity
sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)

rMatrix = model.rMatrix()
res = torch.matmul(rMatrix.transpose(1, 0), torch.matmul(sym, rMatrix)) - sym
print("sym penalty before training: {}".format(torch.norm(res)))

# activate gradients on kick maps
for m in model.modules():
    if type(m) is Elements.Quadrupole:
        for mod in m.modules():
            if type(mod) is Maps.QuadKick:
                mod.requires_grad_(True)

# set up optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# train loop
epochs = int(2e2)
for epoch in range(epochs):
    optimizer.zero_grad()

    out = model(bunch, outputPerElement=outputPerElement)
    loss = criterion(out, label)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 19:
        print(loss.item())

# plot final trajectories
tools.plot.trajectories(axes[2], tools.plot.track(model, bunch, 1), model)
axes[2].set_ylabel("trained")

axes[2].set_xlabel("pos / m")

plt.show()
plt.close()

# test symplecticity
sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)

rMatrix = model.rMatrix()
res = lambda x: torch.matmul(x.transpose(1, 0), torch.matmul(sym, x)) - sym

print("sym penalty after training: {}".format(torch.norm(res(rMatrix))))

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
