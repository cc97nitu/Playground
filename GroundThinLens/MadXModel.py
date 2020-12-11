import torch
import numpy as np

from cpymad.madx import Madx

import ThinLens.Models


class MadXModel(object):
    def __init__(self, model: ThinLens.Models.Model):
        self.model = model

        # get sequence
        self.sequence = self.model.madX()

        # identify terminal maps
        self.terminalMaps = list()

        identifier = 0
        for element in self.model.elements:
            for m in element.maps:
                identifier += 1

            self.terminalMaps.append("map{}".format(identifier))

        return

    def track(self, bunch: torch.tensor):
        numParticles = len(bunch)
        dim = len(bunch[0])
        dtype = bunch.dtype

        # create madx instance
        madx = Madx(stdout=False)

        # specify beam <- properties not important?
        madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=1.012291786)  # from Adrian

        # activate sequence
        madx.input(self.sequence)
        madx.command.use(sequence="SIS18")

        # track
        madx.command.track(onepass=True, onetable=True, )

        for particle in bunch:
            if dim == 4:
                madx.command.start(x=particle[0].item(), px=particle[1].item(), y=particle[2].item(),
                                   py=particle[3].item())
            elif dim == 6:
                madx.command.start(x=particle[0].item(), px=particle[1].item(), y=particle[2].item(),
                                   py=particle[3].item(), t=particle[4].item(), pt=particle[5].item())
            else:
                raise NotImplementedError("bunch dimension not feasible")

        for mapName in self.terminalMaps:
            madx.command.observe(place=mapName)

        madx.command.run(turns=1)
        madx.command.endtrack()

        # merge tracking results
        trackResults = madx.table.trackone

        if dim == 4:
            result = np.stack([trackResults.x, trackResults.px, trackResults.y, trackResults.py, ])
            result = np.reshape(result, (4, -1, numParticles))  # dim, element, particle
            result = np.transpose(result, (2, 0, 1))  # particle, dim, element
        elif dim == 6:
            result = np.stack(
                [trackResults.x, trackResults.px, trackResults.y, trackResults.py, trackResults.t, trackResults.pt])
            result = np.reshape(result, (6, -1, numParticles))  # dim, element, particle
            result = np.transpose(result, (2, 0, 1))  # particle, dim, element

        return torch.as_tensor(result, dtype=dtype)[:,:,1:]  # exclude initial bunch

    def twissTable(self):
        # create madx instance
        madx = Madx(stdout=False)

        # specify beam
        madx.command.beam(mass=18.798, charge=7, exn=1.258e-6, eyn=2.005e-6, gamma=1.012291786)  # from Adrian

        # activate sequence
        madx.input(self.sequence)
        madx.command.use(sequence="SIS18")

        # run twiss module
        twiss = madx.twiss()

        return twiss

    def tune(self):
        twiss = self.twissTable()

        # get tune of one-turn
        tuneX = twiss["mux"][-1]
        tuneY = twiss["muy"][-1]

        return [tuneX, tuneY]


if __name__ == "__main__":
    from ThinLens.Models import SIS18_Lattice_minimal

    # define bunch
    dtype = torch.double

    bunch = torch.tensor([[-1e-2, 0, 2e-3, 0], [1e-3, -1e-3, 1e-2, 0],], dtype=dtype)

    slices = 1

    # load thin lens model
    print("thin lens model:")
    model = SIS18_Lattice_minimal(slices=slices, quadSliceMultiplicity=1, dtype=dtype)
    madXModel = MadXModel(model)

    trackResults = madXModel.track(bunch,)
    print(trackResults.shape)

    tlTrackResults = model(bunch, outputPerElement=True)
    print(tlTrackResults.shape)

    print("diff: {}".format(torch.norm(trackResults[:,:,-1] - tlTrackResults[:,:,-1])))
