'''Gupta potential'''

from autograd import elementwise_grad as egrad
import autograd.numpy as np
from numpy import linalg as la
from scipy.optimize import minimize

# Gupta parameters Fe-Fe, Fe-Co, Fe-Ni, Co-Co, Co-Fe, Co-Ni, Ni-Ni, Ni-Fe, Ni-Co, Pd-Pd, Pd-Pt, Pt-Pt, Pt-Pd
COHESIVE_ENERGY_A = {('Fe', 'Fe'): 0.13315, ('Fe', 'Co'): 0.11246, ('Fe', 'Ni'): 0.07075,
                                ('Co', 'Co'): 0.09500, ('Co', 'Fe'): 0.11246, ('Co', 'Ni'): 0.05970,
                                ('Ni', 'Ni'): 0.03760, ('Ni', 'Fe'): 0.07075, ('Ni', 'Co'): 0.05970,
                                ('Pd', 'Pd'): 0.17460, ('Pd', 'Pt'): 0.2300,
                                ('Pt', 'Pt'): 0.29750, ('Pt', 'Pd'): 0.2300}

COHESIVE_ENERGY_XI = {('Fe', 'Fe'): 1.6179, ('Fe', 'Co'): 1.5515, ('Fe', 'Ni'): 1.3157,
                                 ('Co', 'Co'): 1.4880, ('Co', 'Fe'): 1.5515, ('Co', 'Ni'): 1.2618,
                                 ('Ni', 'Ni'): 1.0700, ('Ni', 'Fe'): 1.3157, ('Ni', 'Co'): 1.2618,
                                 ('Pd', 'Pd'): 1.7180, ('Pd', 'Pt'): 2.2000,
                                 ('Pt', 'Pt'): 2.6950, ('Pt', 'Pd'): 2.2000}

INDEPENT_ELASTIC_CONSTANTS_P = {('Fe', 'Fe'): 10.500, ('Fe', 'Co'): 11.0380, ('Fe', 'Ni'): 13.3599,
                                ('Co', 'Co'): 11.604, ('Co', 'Fe'): 11.0380, ('Co', 'Ni'): 14.0447,
                                ('Ni', 'Ni'): 16.999, ('Ni', 'Fe'): 13.3599, ('Ni', 'Co'): 14.0447,
                                ('Pd', 'Pd'): 10.867, ('Pd', 'Pt'): 10.740,
                                ('Pt', 'Pt'): 10.612, ('Pt', 'Pd'): 10.740}

INDEPENT_ELASTIC_CONSTANTS_Q = {('Fe', 'Fe'): 2.6000, ('Fe', 'Co'): 2.4379, ('Fe', 'Ni'): 1.7582,
                                ('Co', 'Co'): 2.2860, ('Co', 'Fe'): 2.4379, ('Co', 'Ni'): 1.6486,
                                ('Ni', 'Ni'): 1.1890, ('Ni', 'Fe'): 1.7582, ('Ni', 'Co'): 1.6486,
                                ('Pd', 'Pd'): 3.7420, ('Pd', 'Pt'): 3.8700,
                                ('Pt', 'Pt'): 4.0040, ('Pt', 'Pd'): 3.8700}

LATTICE_PARAMETERS_R0 = {('Fe', 'Fe'): 2.5530, ('Fe', 'Co'): 2.5248, ('Fe', 'Ni'): 2.5213,
                         ('Co', 'Co'): 2.4970, ('Co', 'Fe'): 2.5248, ('Co', 'Ni'): 2.4934,
                         ('Ni', 'Ni'): 2.4900, ('Ni', 'Fe'): 2.5213, ('Ni', 'Co'): 2.4934,
                         ('Pd', 'Pd'): 2.7485, ('Pd', 'Pt'): 2.7600,
                         ('Pt', 'Pt'): 2.7747, ('Pt', 'Pd'): 2.7600}

global FILE_PATH_IN
global FILE_PATH_OUT

# Optimizer
def optimize():
    atoms, coord = read_file_xyz()
    x0 = np.array(coord).astype(np.float64)
    x0 = x0.flatten()

    print("Optimizing...")
    
    # Scipy library minimization of scalar function using the BFGS algorithm
    sol = minimize(potential, x0, method='BFGS', jac=grad, options={'gtol':1e-8, 'disp':True})
    x = sol.x

    write_new_file_xyz(x, FILE_PATH_OUT)

    return potential(x), x

# Read the file .xyz
def read_file_xyz():
    atoms = []
    coordinates = []
    with open(FILE_PATH_IN, 'r') as file_object:
        file_object.readline()
        file_object.readline()
        for line in file_object:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])

    return atoms, coordinates

# Write a new file in .xyz format
def write_new_file_xyz(x, file):
    atoms, coordinates = read_file_xyz()  
    n_atoms = len(coordinates)
    with open(file, 'w') as file_object:
        file_object.write(f'{int(n_atoms)}\n')
        file_object.write(f'E = {potential(x)}, |proj g| = {la.norm(grad(x))}\n')
        i = 0
        j = 1
        k = 2
        for n in range(n_atoms):
            file_object.write('{}   {: .8f}       {: .8f}       {: .8f}\n'.format(atoms[n], x[i], x[j], x[k]))
            i += 3
            j += 3
            k += 3

# Gives the parameters values from the dictionary
def parameters():
    atoms, coordinates = read_file_xyz()  
    n_atoms = len(coordinates)

    A = np.zeros((n_atoms, n_atoms))
    XI = np.zeros((n_atoms, n_atoms))
    P = np.zeros((n_atoms, n_atoms))
    Q = np.zeros((n_atoms, n_atoms))
    R0 = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            A[i, j] = COHESIVE_ENERGY_A[(atoms[i], atoms[j])]
            XI[i, j] = COHESIVE_ENERGY_XI[(atoms[i], atoms[j])]
            P[i, j] = INDEPENT_ELASTIC_CONSTANTS_P[(atoms[i], atoms[j])]
            Q[i, j] = INDEPENT_ELASTIC_CONSTANTS_Q[(atoms[i], atoms[j])]
            R0[i, j] = LATTICE_PARAMETERS_R0[(atoms[i], atoms[j])]
    
    return A, XI, P, Q, R0

# Gupta Potential
'''
Raju P. Gupta
Lattice relaxation at a metal surface
Phys. Rev. B 23, 6265 - Published 15 June 1981
https://doi.org/10.1103/PhysRevB.23.6265
'''
def potential(x):
    A, XI, P, Q, R0 = parameters()
    atoms, _ = read_file_xyz()
    n_atoms = len(atoms)
    x = x.reshape(n_atoms, 3)
    U = 0
    for i in range(n_atoms):
        Ub = 0
        Ur = 0
        for j in range(n_atoms):
            if j != i:
                Ub += (XI[i, j]**2) * np.exp(-2 * Q[i, j] * ((np.sqrt((x[i, 0]-x[j, 0])**2 + (x[i, 1]-x[j, 1])**2 +(x[i, 2]-x[j, 2])**2) / R0[i, j]) - 1))
                Ur += A[i, j] * np.exp(-P[i, j] * ((np.sqrt((x[i, 0]-x[j, 0])**2 + (x[i, 1]-x[j, 1])**2 +(x[i, 2]-x[j, 2])**2) / R0[i, j]) - 1))
            else:
                pass
        U += (Ur - np.sqrt(Ub)) 
    return U

# Calculate the gradient
def grad(x):
    df = egrad(potential)
    return df(x)
