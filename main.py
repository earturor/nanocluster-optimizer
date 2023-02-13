# This is a program to optimize Nanoclusters
# Arturo Rentería
# email: earturordr@outlook.com

from potentials import gupta
from packages import rnd_cluster

def main():
    # Generate a random cluster
    file_path_out = './outputs/test-rnd.xyz'
    atoms = {'Fe':2, 'Co':3, 'Ni':4}
    rnd_cluster.random_cluster(file_path_out, atoms)

    gupta.FILE_PATH_IN = file_path_out
    gupta.FILE_PATH_OUT = './outputs/test-rnd-opt.xyz'
    energy, x = gupta.optimize()
    print("Energy", energy)

'''
    # Type the path of the cluster
    gupta.FILE_PATH_IN = './inputs/test.xyz'
    gupta.FILE_PATH_OUT = './outputs/test-opt.xyz'
    
    # Optimize the cluster
    energy, x = gupta.optimize()
    print("Energy:", energy)
'''
    

if __name__ == '__main__':
    main()
