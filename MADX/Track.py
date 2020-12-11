import torch
import cpymad

print(cpymad.__file__)





if __name__ == "__main__":
    import numpy as np

    import tools.madX
    import tools.tuneFFT

    from ThinLens.Models import SIS18_Lattice_minimal

    # define bunch
    dtype = torch.double

    bunch = torch.tensor([[-1e-2, 0, 2e-3, 0], [1e-3, -1e-3, 1e-2, 0],], dtype=dtype)

    # load SIS18
    sequence = open("res/SIS18_Lattice_minimal.mad", "r").read()
    slices = 1

    # load thin lens model
    print("thin lens model:")
    model = SIS18_Lattice_minimal(slices=slices, quadSliceMultiplicity=1, dtype=dtype)
    sequence = model.madX()

    trackResults = tools.madX.track(bunch, sequence,)
    print(trackResults.shape)

    tlTrackResults = model(bunch, outputPerElement=True)
    print(tlTrackResults.shape)

    # try fft

    tunes = tools.tuneFFT.getTuneChromaticity(model, 100, dtype)
    print(tunes)



