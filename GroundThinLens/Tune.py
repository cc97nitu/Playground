import torch
import ThinLens
from ThinLens.Models import SIS18_Lattice_minimal, F0D0Model, SIS18_Lattice
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

    slices = 10
    quadSliceMultiplicity = 1

    model = SIS18_Lattice(slices=slices, quadSliceMultiplicity=quadSliceMultiplicity, dim=dim)
    perturbedModel = SIS18_Lattice_minimal(k1f=k1f, k1d=k1d, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity)
    # model = F0D0Model(k1=k1f, dim=6, dtype=dtype)
    # perturbedModel = F0D0Model(k1=k1d, dim=6, dtype=dtype)

    with open("/dev/shm/second.json", "r") as file:
        model.loadJSON(file)

    # tunes from Mad-X
    madXTunes = tools.madX.tune(model.madX())
    # madXTunes = tools.madX.tune(model.thinMultipoleMadX())
    print(madXTunes)

    madXModel = MadXModel(model)

    # # print tunes
    # tunes = model.getTunes()
    # perturbedTunes = perturbedModel.getTunes()
    #
    # print("model:", tunes, "perturbed model", perturbedTunes)
    # print("tune diff: x={:.2e}, y={:.2e}".format(perturbedTunes[0] - tunes[0], perturbedTunes[1] - tunes[1]))
    # print("madX:", madXModel.tune())

    # # try fft
    # beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e1))
    #
    # tunesFft = tools.tuneFFT.getTuneChromaticity(model, turns=1000, dtype=dtype, beam=beam)
    # print(tunesFft)
    # tools.tuneFFT.getTune(model, turns=1000, dtype=dtype, beam=beam)

