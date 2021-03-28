import sys

sys.path.append("/home/conrad/ThesisWorkspace/Tracking/Playground/GroundThinLens")

import torch
import cpymad
from cpymad.madx import Madx

from ThinLens.Models import SIS18_Lattice, SIS18_Lattice_minimal

from SampleBeam import Beam

# create madx instance
madx = Madx(stdout=False)

# set up beam
beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.000, particles=500)
madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=19.0 / 18.798)  # from Adrian

# activate sequence
Lattice = SIS18_Lattice_minimal
madx.input(Lattice().thinMultipoleMadX())
madx.command.use(sequence="SIS18")

# track
madx.command.track(onepass=True, onetable=True, )

for particle in beam.bunch:
    madx.command.start(x=particle[0].item(), px=particle[1].item(), y=particle[2].item(), py=particle[3].item(),
                       t=particle[4].item(), pt=particle[5].item())

madx.command.dynap(turns=624)
madx.command.endtrack()

dynap = madx.table.dynap
