import time

import numpy as np
import torch

from ThinLens.Models import SIS18_Lattice_minimal

from SampleBeam import Beam


# set up model
device = torch.device("cpu")
dtype = torch.float32
dim = 4

model = SIS18_Lattice_minimal(dim=dim, slices=4, dtype=dtype).to(device)

# load bunch
print("loading bunch")
if dim == 4:
    bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
    bunch = torch.as_tensor(bunch, dtype=dtype)[:2000,:4]
else:
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(2e3))
    bunch = beam.bunch
    print(bunch.shape)

bunch = bunch.to(device)

# track
turns = 10
print("started tracking")
t0 = time.time()

x = bunch
with torch.no_grad():
    for i in range(turns):
        x = model(x)

        if i % 50 == 49:
            print("{}%".format((i + 1) / turns * 100),)

# model(bunch)

print("tracking completed within {:.2f}".format(time.time() - t0))
