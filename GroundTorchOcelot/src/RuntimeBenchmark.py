import time
import numpy as np
import torch

from Simulation.Lattice import SIS18_Lattice_minimal
from Simulation.Models import SecondOrderModel


def track(model, bunch, turns: int):

    with torch.no_grad():
        x = bunch.to(device)

        t0 = time.time()

        for turn in range(turns):
            x = model(x)

        t = time.time() - t0

    return t


if __name__ == "__main__":
    # choose device
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on {}".format(str(device)))

    # create model of SIS18
    print("building model")
    dim = 6
    lattice = SIS18_Lattice_minimal(nPasses=1)
    model = SecondOrderModel(lattice, dim, dtype=dtype)
    model.to(device)

    # load bunch
    print("loading bunch")
    bunch = np.loadtxt("../res/bunch_6d_n=1e6.txt.gz")
    bunch = torch.as_tensor(bunch, dtype=dtype)
    bunch.to(device)

    # track
    turns = 100

    print("started tracking")
    model(bunch[:100])

    particles = [10**i for i in range(1,3)]
    benchmark = [[n, track(model, bunch[:n], turns,)] for n in particles]

    benchmark = np.array(benchmark)
    print(benchmark)

    # dump results
    np.savetxt("../dump/runtimeBenchmark_{}_turns={}_v2.npy".format(device, turns), benchmark)
