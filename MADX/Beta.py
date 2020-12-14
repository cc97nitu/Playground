import torch
import cpymad

import matplotlib.pyplot as plt

from ThinLens.Models import Model
import tools.madX

print(cpymad.__file__)


def plotBeta(sequence: str, ax: plt.axes):
    # get twiss table from Mad-X
    twiss = tools.madX.twissTable(sequence,)

    # plot
    ax.plot(twiss.s, twiss.betx, label="betx")
    ax.plot(twiss.s, twiss.bety, label="bety")

    # print("initial twiss: {}".format([twiss.betx[0], twiss.alfx[0],]))
    # print("final twiss: {}".format([twiss.betx[-1], twiss.alfx[-1],]))
    # print(twiss)

    return ax


def twissMatrix(rMatrix: torch.tensor):
    # consider x-plane
    c, s, cp, sp = rMatrix[0,0], rMatrix[0,1], rMatrix[1,0], rMatrix[1,1]

    twissX = torch.tensor([[c**2, -2*s*c, s**2], [-1*c*cp, s*cp+sp*c, -1*s*sp], [cp**2, -2*sp*cp, sp**2],], dtype=rMatrix.dtype)

    # y-plane
    c, s, cp, sp = rMatrix[2, 2], rMatrix[2, 3], rMatrix[3, 2], rMatrix[3, 3]

    twissY = torch.tensor(
        [[c ** 2, -2 * s * c, s ** 2], [-1 * c * cp, s * cp + sp * c, -1 * s * sp], [cp ** 2, -2 * sp * cp, sp ** 2], ],
        dtype=rMatrix.dtype)

    return twissX, twissY


if __name__ == "__main__":
    import numpy as np
    from ThinLens.Models import SIS18_Cell_minimal, SIS18_Lattice_minimal

    torch.set_printoptions(precision=3,)
    # np.set_printoptions(precision=3, suppress=True)


    # set up model
    model = SIS18_Cell_minimal(slices=8)

    # calculate twiss along ring?
    betx0, alfx0 = 13.502897137826077, 1.6894775514108238
    twiss = [np.array([betx0, alfx0, (1 + alfx0**2)/betx0])]

    for element in model.elements:
        twissM = twissMatrix(element.rMatrix())[0].numpy()
        twiss.append(np.dot(twissM, twiss[-1]))

    # is initial twiss eigenvector?
    twiss = np.stack(twiss)
    print("is initial twiss eigenvector?")
    diff = np.dot(twissMatrix(model.rMatrix())[0].numpy(), twiss[0],) - twiss[0]
    print(diff)
    print(np.allclose(diff, np.zeros(3), atol=1e-4))

    twiss = twiss.transpose()

    # show beta
    fig, ax = plt.subplots()
    ax = plotBeta(model.madX(), ax)

    ax.plot(model.endPositions, twiss[0, 1:])

    # add units
    ax.set_xlabel("pos / m")
    ax.set_ylabel("m")

    plt.legend()

    plt.show()
    plt.close()

