import tools.madX


if __name__ == "__main__":
    from ThinLens.Models import SIS18_Cell_minimal, SIS18_Lattice_minimal

    slices = 10
    Lattice = SIS18_Cell_minimal

    # # load SIS18
    # sequence = open("res/SIS18_Lattice_minimal.mad", "r").read()
    # madXTwiss = getTwiss(sequence, slices=slices)

    # load thin lens model
    print("thin lens model:")

    k1d = -4.78047e-01
    k1f: float = 3.12391e-01
    perturbation = 1.05
    # perturbation = 1.1572

    model = Lattice(k1f=k1f, k1d=k1d, slices=slices, quadSliceMultiplicity=1)
    perturbedModel = Lattice(k1f=perturbation*k1f, k1d=k1d, slices=slices, quadSliceMultiplicity=1)

    print("model tunes: {}".format(model.getTunes()))
    print("perturbed model tunes: {}".format(perturbedModel.getTunes()))

    # get tunes from Mad-X
    print("\nMad-X")

    twiss = tools.madX.twissTable(model.madX(), slices=slices)
    perturbedTwiss = tools.madX.twissTable(perturbedModel.madX(), slices=slices)

    # calculate twiss difference
    tuneXDiff = perturbedTwiss["mux"][-1] - twiss["mux"][-1]
    tuneYDiff = perturbedTwiss["muy"][-1] - twiss["muy"][-1]

    print("model tunes: {}".format([twiss["mux"][-1], twiss["muy"][-1],]))
    print("perturbed model tunes: {}".format([perturbedTwiss["mux"][-1], perturbedTwiss["muy"][-1],]))
    print("tune differences: x={:.2e}, y={:.2e}".format(tuneXDiff, tuneYDiff))
