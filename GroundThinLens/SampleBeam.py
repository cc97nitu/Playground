import math

import torch
import torch.distributions


class Beam(object):
    def __init__(self, mass: float, energy: float, exn: float, eyn: float, sigt: float, sige: float, particles: int, centroid: list = (0,0,0,0,0)):
        """
        Set up beam as Mad-X beam command does.
        """
        # calculate properties of reference particle
        self.energy = energy  # GeV
        self.mass = mass  # GeV

        self.gamma = self.energy / self.mass
        self.momentum = math.sqrt(self.energy ** 2 - self.mass ** 2)  # GeV/c

        self.beta = self.momentum / (self.gamma * self.mass)

        # standard deviations assuming round beams
        ex = exn / (self.beta * self.gamma)  # m
        ey = eyn / (self.beta * self.gamma)  # m

        stdX = math.sqrt(ex / math.pi)
        stdY = math.sqrt(ey / math.pi)

        stdE = sige * self.energy  # GeV

        std = torch.FloatTensor([stdX, stdX, stdY, stdY, sigt, stdE])

        # sample particles
        loc = torch.FloatTensor([*centroid, self.energy])
        scaleTril = torch.diag(std**2)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=scaleTril)

        preliminaryBunch = dist.sample((particles,))  # x, xp, y, yp, sigma, totalEnergy

        # calculate missing properties for individual particles
        pSigma = (preliminaryBunch[:,5] - self.energy) / (self.beta * self.momentum)

        momentum = torch.sqrt(preliminaryBunch[:,5]**2 - self.mass**2)

        delta = (momentum - self.momentum) / self.momentum
        invDelta = 1 / (delta + 1)
        gamma = preliminaryBunch[:,5] / self.mass
        beta = momentum / (gamma * self.mass)
        velocityRatio = self.beta / beta

        # create bunch
        x = preliminaryBunch[:,0]
        xp = preliminaryBunch[:,1]
        y = preliminaryBunch[:,2]
        yp = preliminaryBunch[:,3]
        sigma = preliminaryBunch[:,4]

        self.bunch = torch.stack([x, xp, y, yp, sigma, pSigma, delta, invDelta, velocityRatio]).t()
        return

if __name__ == "__main__":
    torch.set_printoptions(precision=1)

    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e6))

    print(beam.beta)
    print(beam.bunch.isnan().any())

