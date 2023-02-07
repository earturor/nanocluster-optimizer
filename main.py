# This is a program to optimize Nanoclusters
# Arturo Rentería
# email: earturordr@outlook.com

from potentials import gupta

def main():
    # Type the path of the cluster
    gupta.FILE_PATH_IN = './inputs/test.xyz'
    gupta.FILE_PATH_OUT = './outputs/test-opt.xyz'
    
    # Optimize the cluster
    energy, x = gupta.optimize()
    print("Energy:", energy)

    # The file path of the optimize cluster is in ./outputs/cluster-opt.xyz

if __name__ == '__main__':
    main()
