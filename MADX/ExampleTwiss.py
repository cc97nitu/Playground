from cpymad.madx import Madx

import cpymad
print(cpymad.__file__)

# create madx instance
madx = Madx()

# specify beam
madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=1.012291786)

# load a sequence
sequence = open("res/SIS18_Lattice_minimal.mad", "r").read()
madx.input(sequence)

madx.command.use(sequence="SIS18")

# run twiss module
madx.command.twiss()
