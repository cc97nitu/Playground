import time

import torch


def track(model, bunch, turns: int):
    device = next(model.parameters()).device
    bunch.to(device)

    # track
    with torch.no_grad():
        t0 = time.time()

        multiTurnOutput = list()
        y = bunch
        for i in range(turns):
            # y = model(y)
            # multiTurnOutput.append(y)

            y = model(y, outputPerElement=True)
            multiTurnOutput.append(y)
            y = y[:, :, -1]

    # prepare tracks for plotting
    trackResults = torch.cat(multiTurnOutput, 2)  # indices: particle, dim, element
    # trackResults = multiTurnOutput[0]  # restrict to first turn, indices: element, particle, dim
    return trackResults


def trajectories(ax, trackResults, lattice):
    """Plot individual trajectories."""
    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    for particle in trackResults:
        # x-coord
        ax.plot(pos, particle[0])

    return


def beamCentroid(ax, trackResults, lattice):
    """Plot beam centroid."""
    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    trackResults = trackResults.permute((1, 2, 0))  # dim, element, particle
    beamCentroid = torch.mean(trackResults, dim=2)

    ax.plot(pos, beamCentroid[0].to("cpu").numpy())
    return


def beamSigma(ax, trackResults, lattice):
    """Plot beam size as standard deviation of position."""
    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    trackResults = trackResults.permute((1, 2, 0))  # dim, element, particle
    beamSigma = torch.std(trackResults, dim=2)

    # plt.plot(pos, beamSigma[0].numpy())
    # plt.show()
    # plt.close()
    ax.plot(pos, beamSigma[0].to("cpu").numpy())
    return
