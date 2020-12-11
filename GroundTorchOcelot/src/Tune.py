import numpy as np
import matplotlib.pyplot as plt

import torch.fft

from TorchOcelot.Lattice import SIS18_Lattice_minimal, SIS18_Lattice
from TorchOcelot.Models import LinearModel, SecondOrderModel


if __name__ == "__main__":
    import tools.tuneFFT

    # create model of SIS18
    dim = 6
    dtype = torch.float32

    lattice = SIS18_Lattice(nPasses=1)
    model = SecondOrderModel(lattice, dim, dtype=dtype)
    turns = 100

    # obtain chroma and tune
    xFit, yFit = tools.tuneFFT.getTuneChromaticity(model, turns, dtype)
    print(xFit, yFit)

    # plot spectrum
    dp = np.linspace(-5e-3, 5e-3, 9)
    bunch = torch.tensor([[1e-2,0,1e-2,0,0,i] for i in dp], dtype=dtype)

    frequencies, fft = tools.tuneFFT.getTuneFFT(bunch, model, turns)
    for f in fft:
        plt.plot(frequencies, f[0])

    plt.show()
    plt.close()

