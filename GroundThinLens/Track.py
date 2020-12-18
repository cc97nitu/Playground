import torch

from ThinLens.Models import SIS18_Cell_minimal, F0D0Model, RBendLine

from SampleBeam import Beam


# general settings
torch.set_printoptions(precision=4, sci_mode=False)
dtype = torch.double
dim = 4

# setup model
# model = RBendLine(angle=0.1, e1=0, e2=0.0, dim=dim, dtype=dtype)
model = SIS18_Cell_minimal(dim=dim, dtype=dtype)

# particles
if dim == 4:
    bunch = torch.tensor([[1e-2, 0, 1e-2, 0], ], dtype=dtype)
else:
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.00, particles=int(1e3))
    bunch = beam.bunch[:3]
    bunch[:,0:4] = torch.tensor([1e-2, 0, 1e-2, 0], dtype=dtype)
    # bunch.unsqueeze_(0)

print("initial bunch")
print(bunch)

# track
res = model(bunch, outputPerElement=True)

print("final bunch")
print(res)
