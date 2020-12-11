import torch
import ThinLens
from ThinLens.Models import SIS18_Lattice_minimal
from MadXModel import MadXModel

if __name__ == "__main__":
    # instantiate models
    k1f = 3.17877e-01
    k1d = -4.76814e-01

    slices = 10
    quadSliceMultiplicity = 4

    model = SIS18_Lattice_minimal(k1f=k1f, k1d=k1d, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity)
    perturbedModel = SIS18_Lattice_minimal(k1f=k1f, k1d=k1d, slices=slices, quadSliceMultiplicity=quadSliceMultiplicity)
    madXModel = MadXModel(model)

    # print tunes
    tunes = model.getTunes()
    perturbedTunes = perturbedModel.getTunes()

    print("model:", tunes, "perturbed model", perturbedTunes)
    print("tune diff: x={:.2e}, y={:.2e}".format(perturbedTunes[0] - tunes[0], perturbedTunes[1] - tunes[1]))
    print("madX:", madXModel.tune())

