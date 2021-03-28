<<<<<<< HEAD
import json
=======
import time
>>>>>>> a92fecf665033427291e2e8b67ae788f04132016
import torch

from ThinLens.Models import SIS18_Lattice, SIS18_Cell_minimal, F0D0Model, RBendLine

from SampleBeam import Beam


# general settings
torch.set_printoptions(precision=4, sci_mode=False)
dtype = torch.double
dim = 6

# setup model
model = SIS18_Lattice(dim=dim, dtype=dtype).requires_grad_(False)

# particles
if dim == 4:
    bunch = torch.tensor([[1e-2, 0, 1e-2, 0], ], dtype=dtype)
else:
<<<<<<< HEAD
    particles = int(1e1)
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.00, particles=particles)
    bunch = beam.bunch
    bunch[:,0] = torch.linspace(-0.1, 0.1, particles)
    bunch[:,1] = torch.linspace(-0.1, 0.1, particles)
    bunch[:,2] = torch.linspace(-0.1, 0.1, particles)
    bunch[:,3] = torch.linspace(-0.1, 0.1, particles)
    # bunch.unsqueeze_(0)

print("initial bunch")
print(bunch)

# track
for turn in range(10):
=======
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.00, particles=int(5e3))
    bunch = beam.bunch

# track
t0 = time.time()

for turn in range(50):
>>>>>>> a92fecf665033427291e2e8b67ae788f04132016
    bunch = model(bunch, outputPerElement=False)

res = model(bunch, outputPerElement=False)

<<<<<<< HEAD
print("final bunch")
print(res)

with open("/dev/shm/bunch.json", "w") as file:
    json.dump(bunch.tolist(), file)
=======
print("tracking completed within {:2f}s".format(time.time() - t0))
>>>>>>> a92fecf665033427291e2e8b67ae788f04132016
