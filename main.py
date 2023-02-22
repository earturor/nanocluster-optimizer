# This is a program to optimize Nanoclusters
# Arturo Rentería
# email: earturordr@outlook.com

import pandas as pd
from potentials import gupta
from packages import rnd_cluster

def main():
    # Generates a certain number of random clusters and subsequently optimizes them.
    epochs = 1 # Type the number of cluster to be generated
    energies = []
    files = []
    for i in range(epochs):
        # Generate a random cluster
        file_path_out = './outputs/test-rnd.xyz'
        atoms = {'Fe':0, 'Co':0, 'Ni':3} # Type the elements and their quantities
        rnd_cluster.random(file_path_out, atoms)
        # Specifies the input and output path and then optimize the cluster
        gupta.FILE_PATH_IN = file_path_out
        gupta.FILE_PATH_OUT = f'./outputs/test-rnd-opt{i}.xyz'
        energy, x = gupta.optimize()
        print("Energy", energy)
        # Create a energy table in csv
        energies.append(float(energy))
        files.append(str(f'test-rnd-opt{i}.xyz'))
        df = pd.DataFrame({ "File": files, "Energy": energies })
        df.to_csv('./outputs/energy_table.csv')

'''
    # Optimize a cluster from a specific path
    gupta.FILE_PATH_IN = './inputs/test.xyz'
    gupta.FILE_PATH_OUT = './outputs/test-opt.xyz'    
    energy, x = gupta.optimize()
    print("Energy:", energy)
'''
    

if __name__ == '__main__':
    main()
