import torch
import ThinLens
from ThinLens.Models import SIS18_Cell_minimal, SIS18_Lattice_minimal, F0D0Model, SIS18_Lattice, RBendLine
from MadXModel import MadXModel

import tools.tuneFFT
import tools.madX
from SampleBeam import Beam


if __name__ == "__main__":
    # instantiate models
    dtype = torch.float32
    dim = 6

    k1f = 3.17877e-01
    k1d = -4.76814e-01

    slices = 4
    quadSliceMultiplicity = 1

    model = SIS18_Lattice(slices=slices, quadSliceMultiplicity=quadSliceMultiplicity, dim=dim)
    model = SIS18_Cell_minimal(slices=slices, quadSliceMultiplicity=quadSliceMultiplicity, dim=dim)
    # model = F0D0Model(k1=0.3, slices=slices, dim=dim)
    # model = SIS18_Cell_minimal(slices=slices, quadSliceMultiplicity=quadSliceMultiplicity, dim=dim)

    # get tune from trace
    print("tunes: {}".format(model.getTunes()))

    # tunes from Mad-X
    madXTunes = tools.madX.tune(model.madX())
    print(madXTunes)

    print("tunes from thin-mulitpole Mad-X: {}".format(tools.madX.tune(model.thinMultipoleMadX())))

