import numpy as np
import matplotlib.pyplot as plt

import torch

from ThinLens.Models import SIS18_Lattice_minimal


if __name__ == "__main__":
    import tools.plot

    # create model of SIS18
    print("building model")
    dim = 6
    dtype = torch.float32
    model = SIS18_Lattice_minimal(dim=dim, dtype=dtype, )
    model.logElementPositions()

    # load bunch
    print("loading bunch")
    bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
    bunch = torch.as_tensor(bunch, dtype=dtype)[:10,:4]
    bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim
    # bunch = bunch + torch.tensor([1e-3, 0, 1e-3, 0, 0, 0], dtype=torch.double)  # bunch has transverse offset

    # visualize accelerator
    trackResults = tools.plot.track(model, bunch, 6)

    fig, axes = plt.subplots(3, sharex=True)
    tools.plot.trajectories(axes[0], trackResults, model)
    tools.plot.beamCentroid(axes[1], trackResults, model)
    tools.plot.beamSigma(axes[2], trackResults, model)

    plt.show()
    plt.close()

    ########################################################################
    bunchCentroid = bunch.permute(1, 0).mean(dim=1)  # dim, particle
    print(bunchCentroid)

    bunchSigma = bunch.permute(1, 0).std(dim=1)  # dim, particle
    print(bunchSigma)