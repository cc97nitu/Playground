import numpy as np
import torch

from cpymad.madx import Madx


def twissTable(sequence: str, slices: int = 0):
    # create madx instance
    madx = Madx(stdout=False)

    # specify beam
    madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=1.012291786)  # from Adrian

    # activate sequence
    madx.input(sequence)
    madx.command.use(sequence="SIS18")

    if slices:
        # make elements thin
        madx.command.select(sequence="SIS18", flag="makethin", slice_=slices)
        madx.command.makethin(sequence="SIS18", style="simple", makedipedge=True)
        madx.command.use(sequence="SIS18")

    # run twiss module
    twiss = madx.twiss()

    return twiss


def tune(sequence: str, slices: int = 0):
    twiss = twissTable(sequence, slices)

    # get tune of one-turn
    tuneX = twiss["mux"][-1]
    tuneY = twiss["muy"][-1]

    return [tuneX, tuneY]


def track(bunch: torch.tensor, sequence, slices: int = 0):
    numParticles = len(bunch)
    dim = len(bunch[0])
    dtype = bunch.dtype

    # create madx instance
    madx = Madx(stdout=False)

    # specify beam
    madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=1.012291786)  # from Adrian

    # activate sequence
    madx.input(sequence)
    madx.command.use(sequence="SIS18")

    if slices:
        print("running make thin")
        # make elements thin
        madx.command.select(sequence="SIS18", flag="makethin", slice_=slices)
        madx.command.makethin(sequence="SIS18", style="simple", makedipedge=True)
        madx.command.use(sequence="SIS18")

    # number of elements?
    sis18 = madx.sequence.sis18

    # track
    madx.command.track(onepass=True, onetable=True,)

    for particle in bunch:
        if dim == 4:
            madx.command.start(x=particle[0].item(), px=particle[1].item(), y=particle[2].item(), py=particle[3].item())
        elif dim == 6:
            madx.command.start(x=particle[0].item(), px=particle[1].item(), y=particle[2].item(), py=particle[3].item(), t=particle[4].item(), pt=particle[5].item())
        else:
            raise NotImplementedError("bunch dimension not feasible")

    for element in list(sis18.elements)[1:-1]:  # exclude 'start' and 'end' markers
        madx.command.observe(place=element.name)

    madx.command.run(turns=1)
    madx.command.endtrack()

    # merge tracking results
    trackResults = madx.table.trackone

    if dim == 4:
        result = np.stack([trackResults.x, trackResults.px, trackResults.y, trackResults.py, ])
        result = np.reshape(result, (4, -1, numParticles))  # dim, element, particle
        result = np.transpose(result, (2, 0, 1))  # particle, dim, element
    elif dim == 6:
        result = np.stack([trackResults.x, trackResults.px, trackResults.y, trackResults.py, trackResults.t, trackResults.pt])
        result = np.reshape(result, (6, -1, numParticles))  # dim, element, particle
        result = np.transpose(result, (2, 0, 1))  # particle, dim, element

    return torch.as_tensor(result, dtype=dtype)


if __name__ == "__main__":
    from ThinLens.Models import SIS18_Cell_minimal

    # track a bunch with Mad-X
    bunch = torch.tensor([[-1e-2, 0, 2e-3, 0], [1e-3, -1e-3, 1e-2, 0], [0,0,0,0]])
    trackResults = track(bunch, SIS18_Cell_minimal().madX())
    print(trackResults)
