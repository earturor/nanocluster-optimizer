# This is a program to optimize Nanoclusters
# Arturo Rentería
# email: earturordr@outlook.com

from argparse import ArgumentParser

from potentials.gupta import GuptaPotential


def main():
    parser = ArgumentParser("Generate optimised nano-clusters")
    parser.add_argument("epochs", type=int)
    parser.add_argument("--fe", type=int, default=0)
    parser.add_argument("--co", type=int, default=0)
    parser.add_argument("--ni", type=int, default=3)
    parser.add_argument("--pd", type=int, default=0)
    parser.add_argument("--pt", type=int, default=0)
    args = parser.parse_args()

    for i in range(args.epochs):
        atoms = {
            "Fe": args.fe,
            "Co": args.co,
            "Ni": args.ni,
            "Pd": args.pd,
            "Pt": args.pt,
        }

        path_src = f"initial-{i:03}.xyz"
        path_out = f"optim-{i:03}.xyz"

        gp, c1 = GuptaPotential.random(atoms)
        e1 = gp.potential(c1)

        gp.write_xyz_file(path_src, c1)

        e2, c2 = gp.optimize(c1)
        gp.write_xyz_file(path_out, c2)

        print(f"{e1:.2f},{e2:.2f},{path_src},{path_out}")


if __name__ == "__main__":
    main()
