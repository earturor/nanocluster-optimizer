"""Gupta potential"""

import autograd.numpy as np
import scipy.optimize as spo
from autograd import elementwise_grad as egrad

# Gupta parameters between Fe-Co-Ni and Pd-Pt
COHESIVE_ENERGY_A = {
    ("Fe", "Fe"): 0.13315,
    ("Fe", "Co"): 0.11246,
    ("Fe", "Ni"): 0.07075,
    ("Co", "Co"): 0.09500,
    ("Co", "Fe"): 0.11246,
    ("Co", "Ni"): 0.05970,
    ("Ni", "Ni"): 0.03760,
    ("Ni", "Fe"): 0.07075,
    ("Ni", "Co"): 0.05970,
    ("Pd", "Pd"): 0.17460,
    ("Pd", "Pt"): 0.2300,
    ("Pt", "Pt"): 0.29750,
    ("Pt", "Pd"): 0.2300,
}

COHESIVE_ENERGY_XI = {
    ("Fe", "Fe"): 1.6179,
    ("Fe", "Co"): 1.5515,
    ("Fe", "Ni"): 1.3157,
    ("Co", "Co"): 1.4880,
    ("Co", "Fe"): 1.5515,
    ("Co", "Ni"): 1.2618,
    ("Ni", "Ni"): 1.0700,
    ("Ni", "Fe"): 1.3157,
    ("Ni", "Co"): 1.2618,
    ("Pd", "Pd"): 1.7180,
    ("Pd", "Pt"): 2.2000,
    ("Pt", "Pt"): 2.6950,
    ("Pt", "Pd"): 2.2000,
}

INDEPENT_ELASTIC_CONSTANTS_P = {
    ("Fe", "Fe"): 10.500,
    ("Fe", "Co"): 11.0380,
    ("Fe", "Ni"): 13.3599,
    ("Co", "Co"): 11.604,
    ("Co", "Fe"): 11.0380,
    ("Co", "Ni"): 14.0447,
    ("Ni", "Ni"): 16.999,
    ("Ni", "Fe"): 13.3599,
    ("Ni", "Co"): 14.0447,
    ("Pd", "Pd"): 10.867,
    ("Pd", "Pt"): 10.740,
    ("Pt", "Pt"): 10.612,
    ("Pt", "Pd"): 10.740,
}

INDEPENT_ELASTIC_CONSTANTS_Q = {
    ("Fe", "Fe"): 2.6000,
    ("Fe", "Co"): 2.4379,
    ("Fe", "Ni"): 1.7582,
    ("Co", "Co"): 2.2860,
    ("Co", "Fe"): 2.4379,
    ("Co", "Ni"): 1.6486,
    ("Ni", "Ni"): 1.1890,
    ("Ni", "Fe"): 1.7582,
    ("Ni", "Co"): 1.6486,
    ("Pd", "Pd"): 3.7420,
    ("Pd", "Pt"): 3.8700,
    ("Pt", "Pt"): 4.0040,
    ("Pt", "Pd"): 3.8700,
}

LATTICE_PARAMETERS_R0 = {
    ("Fe", "Fe"): 2.5530,
    ("Fe", "Co"): 2.5248,
    ("Fe", "Ni"): 2.5213,
    ("Co", "Co"): 2.4970,
    ("Co", "Fe"): 2.5248,
    ("Co", "Ni"): 2.4934,
    ("Ni", "Ni"): 2.4900,
    ("Ni", "Fe"): 2.5213,
    ("Ni", "Co"): 2.4934,
    ("Pd", "Pd"): 2.7485,
    ("Pd", "Pt"): 2.7600,
    ("Pt", "Pt"): 2.7747,
    ("Pt", "Pd"): 2.7600,
}


def _is_symmtric(d: dict[tuple[str, str], float]) -> bool:
    "make sure d has entries for each pair of each set and they're symmetric"
    import itertools

    atomic_sets = [
        ["Fe", "Co", "Ni"],
        ["Pd", "Pt"],
    ]
    for set in atomic_sets:
        for a, b in itertools.product(set, set):
            assert d[(a, b)] == d[(b, a)]
    return True


assert _is_symmtric(COHESIVE_ENERGY_A)
assert _is_symmtric(COHESIVE_ENERGY_XI)
assert _is_symmtric(INDEPENT_ELASTIC_CONSTANTS_P)
assert _is_symmtric(INDEPENT_ELASTIC_CONSTANTS_Q)
assert _is_symmtric(LATTICE_PARAMETERS_R0)


class GuptaPotential:
    def __init__(self, atoms: list[str]) -> None:
        self.atoms = atoms

        n = len(atoms)
        idx_ij = []
        atom_ij = []

        # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html also
        # uses this ordering
        for i in range(n - 1):
            for j in range(i + 1, n):
                idx_ij.append([i, j])
                atom_ij.append((atoms[i], atoms[j]))

        arr_ij = np.array(idx_ij, dtype=int)
        self.ai = arr_ij[:, 0]
        self.aj = arr_ij[:, 1]
        self.A = np.array([COHESIVE_ENERGY_A[aij] for aij in atom_ij], dtype=float)
        self.XI = np.array([COHESIVE_ENERGY_XI[aij] for aij in atom_ij], dtype=float)
        self.P = np.array(
            [INDEPENT_ELASTIC_CONSTANTS_P[aij] for aij in atom_ij], dtype=float
        )
        self.Q = np.array(
            [INDEPENT_ELASTIC_CONSTANTS_Q[aij] for aij in atom_ij], dtype=float
        )
        self.R0 = np.array([LATTICE_PARAMETERS_R0[aij] for aij in atom_ij], dtype=float)

        # calculate these once instead of every time in potential
        self.XI2 = self.XI**2
        self.nP = -self.P
        self.nQ2 = -self.Q * 2

        def idx(i, j):
            "see above scipy link"
            if j < i:
                i, j = j, i
            return n * i + j - ((i + 2) * (i + 1)) // 2

        self.pairwise = np.array(
            [[idx(i, j) for j in range(n) if i != j] for i in range(n)]
        )

        # bind this here so we don't try and take the gradient of self
        self.gradient = egrad(self.potential)

    def potential(self, coords: np.ndarray) -> float:
        """
        Raju P. Gupta
        Lattice relaxation at a metal surface
        Phys. Rev. B 23, 6265 - Published 15 June 1981
        https://doi.org/10.1103/PhysRevB.23.6265
        """
        dist = np.linalg.norm(coords[self.ai] - coords[self.aj], axis=1)
        norm = dist / self.R0 - 1.0
        Ub = self.XI2 * np.exp(self.nQ2 * norm)
        Ur = self.A * np.exp(self.nP * norm)
        U: float = 2.0 * np.sum(Ur)
        U -= np.sum(np.sqrt(np.sum(Ub[self.pairwise], axis=1)))
        return U

    @classmethod
    def random(
        cls, atom_counts: dict[str, int], lb: float = -2.56, ub: float = 2.56
    ) -> tuple["GuptaPotential", np.ndarray]:
        atoms = []
        for atom, count in atom_counts.items():
            atoms.extend([atom] * count)

        coords = np.random.uniform(lb, ub, (len(atoms), 3))
        return cls(atoms), coords

    @classmethod
    def read_xyz_file(
        cls, path: str, *, validate: bool = False
    ) -> tuple["GuptaPotential", np.ndarray]:
        atoms = []
        coordinates = []
        with open(path, "r") as fd:
            num_atoms = fd.readline()
            state = fd.readline()
            for line in fd:
                atom, *xyz = line.split()
                atoms.append(atom)
                coordinates.append(xyz)

        coords = np.array(coordinates, dtype=float)
        gp = cls(atoms)

        if validate:
            if int(num_atoms) != len(coordinates):
                raise ValueError(
                    "n_atoms from file doesn't match number of coordinates"
                )

            e, _ = state.split(",", 1)
            _, e = e.split("=")

            if abs(float(e) - gp.potential(coords)) > 0.0001:
                raise ValueError(
                    "potential stored in file doesn't match calculated value"
                )

        return gp, coords

    def write_xyz_file(self, path: str, coords: np.ndarray) -> None:
        n = len(self.atoms)
        assert coords.shape == (n, 3)

        pot_x = self.potential(coords)
        proj_g = np.linalg.norm(self.gradient(coords))

        with open(path, "w") as fd:
            print(len(self.atoms), file=fd)
            print(f"E = {pot_x}, |proj g| = {proj_g}", file=fd)
            for atom, [x, y, z] in zip(self.atoms, coords):
                print(f"{atom}   {x: .8f}       {y: .8f}       {z: .8f}", file=fd)

    def optimize(
        self, coords: np.ndarray, *, gtol: float = 1e-6, disp: bool = False
    ) -> tuple[float, np.ndarray]:
        n = len(self.atoms)
        assert coords.shape == (n, 3)

        def potential(x: np.ndarray) -> float:
            return self.potential(x.reshape(n, 3))

        # sol = spo.basinhopping(
        #     potential,
        #     coords.flatten(),
        #     minimizer_kwargs={
        #         "method": "BFGS",
        #         "jac": egrad(potential),
        #         },
        #     niter=250
        # )
                   
        sol = spo.minimize(
            potential,
            coords.flatten(),
            method="BFGS",
            jac=egrad(potential),
            options={
                "gtol": gtol,
                "disp": disp,
            },
        )

        return sol.fun, sol.x.reshape(n, 3)


# Optimizer
def optimize(path_in: str, path_out: str) -> tuple[float, np.ndarray]:
    gp, coords = GuptaPotential.read_xyz_file(path_in)

    print("Optimizing...")

    fn, coords = gp.optimize(coords)

    gp.write_xyz_file(path_out, coords)

    return fn, coords
