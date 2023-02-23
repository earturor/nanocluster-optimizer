from pathlib import Path

from potentials.gupta import GuptaPotential


def main():
    for path in Path("test-data").glob("*.xyz"):
        # passing validate=True will cause code to throw if file is inconsistent
        gp, c1 = GuptaPotential.read_xyz_file(path, validate=True)
        e1 = gp.potential(c1)
        e2, c2 = gp.optimize(c1, disp=True)

        print(f"optimising {path} changed energy from {e1:.6f} to {e2:.6f}")


if __name__ == "__main__":
    main()
