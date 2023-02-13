'''Generate a random cluster'''

import numpy as np

def random_cluster(file_path_out, atoms, lb=2.56, ub=-2.56):
    atoms_list = []
    for key in atoms:
        for i in range(atoms[key]):
            atoms_list.append(key)

    n_atoms = len(atoms_list)
    x0 = np.random.uniform(lb, ub, (n_atoms, 3))
    x0 = x0.flatten()
    write_new_file_xyz(x0, file_path_out, atoms_list)

def write_new_file_xyz(x, file, atoms_list):
    n_atoms = len(atoms_list)
    with open(file, 'w') as file_object:
        file_object.write(f'{int(n_atoms)}\n')
        file_object.write('\n')
        i = 0
        j = 1
        k = 2
        for n in range(n_atoms):
            file_object.write('{}   {: .8f}       {: .8f}       {: .8f}\n'.format(atoms_list[n], x[i], x[j], x[k]))
            i += 3
            j += 3
            k += 3