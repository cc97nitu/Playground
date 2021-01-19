import json
import torch

from ThinLens.Models import SIS18_Lattice, SIS18_Cell_minimal, F0D0Model, RBendLine

from SampleBeam import Beam


# general settings
torch.set_printoptions(precision=4, sci_mode=False)
dtype = torch.double
dim = 6

# setup model
# model = F0D0Model(k1=0.3, dtype=dtype)
# model = RBendLine(angle=0.1, e1=0.1, e2=-0.1, dim=dim, dtype=dtype)
model = SIS18_Lattice(dim=dim, dtype=dtype)

# particles
if dim == 4:
    bunch = torch.tensor([[1e-2, 0, 1e-2, 0], ], dtype=dtype)
else:
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
    bunch = model(bunch, outputPerElement=False)

res = model(bunch, outputPerElement=False)

print("final bunch")
print(res)

with open("/dev/shm/bunch.json", "w") as file:
    json.dump(bunch.tolist(), file)
