from potentials import gupta


def main():
    gp, coords = gupta.GuptaPotential.random({"Fe": 3, "Co": 9, "Ni": 7})

    gp.write_xyz_file("start.xyz", coords)

    pot, c2 = gp.optimize(coords, disp=True)

    gp.write_xyz_file("output.xyz", c2)


if __name__ == "__main__":
    main()
