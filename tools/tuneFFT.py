import time
import numpy as np
import torch
import torch.fft

import matplotlib.pyplot as plt


def getTuneFFT(bunch: torch.Tensor, model, turns: int):
    # track
    with torch.no_grad():
        phaseSpace = [bunch, ]
        for turn in range(turns):
            phaseSpace.append(model(phaseSpace[-1]))

    phaseSpace = torch.stack(phaseSpace)  # turn, particle, dim
    phaseSpace = phaseSpace.permute((1,2,0))  # particle, dim, turn
    phaseSpace = phaseSpace[:,[0,2],:]  # keep only x-, and y-coord

    # remove dispersion
    mean = phaseSpace.mean(dim=2).unsqueeze(-1)
    phaseSpace = phaseSpace - mean

    # apply fft
    windowSize = 1 * turns  # zero padding?

    fft = torch.abs(torch.fft.fft(phaseSpace, n=windowSize, dim=2, norm="forward"))

    # calculate frequencies
    sampleRatePerTurn = 1
    frequencies = np.fft.fftfreq(windowSize, sampleRatePerTurn)

    idx = np.argsort(frequencies)
    idx = idx[len(idx)//2:]  # remove negative frequencies

    return frequencies[idx], fft[:,:,idx]


def getTuneChromaticity(model, turns: int, dtype: torch.dtype, beam=None):
    # set up bunch
    dp = torch.linspace(-5e-3, 5e-3, 9, dtype=dtype)

    if not beam:
        # set up particles
        if model.dim == 6:
            bunch = torch.tensor([[1e-2, 0, 1e-2, 0, 0, i] for i in dp], dtype=dtype)
        elif model.dim == 4:
            bunch = torch.tensor([[1e-2, 0, 1e-2, 0], ], dtype=dtype)
        elif model.dim == 2:
            bunch = torch.tensor([[1e-2, 0], ], dtype=dtype)
        else:
            raise RuntimeError("illegal model dimension")
    else:
        # beam provides bunch
        bunch = beam.fromDelta(dp)

    # bring model and bunch to same location
    bunch.to("cpu")
    model.to("cpu")

    # get fft
    frequencies, fft = getTuneFFT(bunch, model, turns)

    # obtain tune and chromaticity from linear fit
    fftArgmax = torch.argmax(fft, dim=2, keepdim=False).transpose(1,0)  # dim, particle

    tunesX = frequencies[fftArgmax[0]]
    tunesY = frequencies[fftArgmax[1]]

    if model.dim == 6:
        xFit = np.polyfit(dp, tunesX, deg=1)
        yFit = np.polyfit(dp, tunesY, deg=1)
        return xFit, yFit
    else:
        return tunesX, tunesY  # all tunes shall be the same


def getTune(model, turns: int, dtype: torch.dtype, beam=None):
    # set up bunch
    dp = torch.linspace(-5e-3, 5e-3, 9, dtype=dtype)

    if not beam:
        # set up particles
        if model.dim == 6:
            bunch = torch.tensor([[1e-2, 0, 1e-2, 0, 0, i] for i in dp], dtype=dtype)
        elif model.dim == 4:
            bunch = torch.tensor([[1e-2, 0, 1e-2, 0], ], dtype=dtype)
        elif model.dim == 2:
            bunch = torch.tensor([[1e-2, 0], ], dtype=dtype)
        else:
            raise RuntimeError("illegal model dimension")
    else:
        # beam provides bunch
        bunch = beam.fromDelta(dp)
        print(bunch.shape)

    # bring model and bunch to same location
    bunch.to("cpu")
    model.to("cpu")

    # get fft
    frequencies, fft = getTuneFFT(bunch, model, turns)

    for particle in range(len(fft)):
        plt.plot(frequencies, fft[particle, 0])

    plt.show()
    plt.close()

