import sys
sys.path.append('..')
import utils.data_gen as dg
import os
import numpy as np
import csv

path_to_networks = '../datasets/Networks/'
cell_lines = os.listdir(path_to_networks)
cell_lines = sorted(cell_lines)

dg.total_exp(cell_lines, path_to_networks)
total = 0

for cl in cell_lines:
    nets = 0
    full_path = path_to_networks + cl + '/'
    list_dir = sorted(os.listdir(path_to_networks + cl))
    total_exp = len(os.listdir(path_to_networks + cl))
    repeat_dir = np.repeat(full_path, total_exp)

    full_dir = [x + y for x,y in zip(repeat_dir, list_dir)]

    sample = full_dir

    for exp in sample:
        if os.path.isdir(exp + '/Results_CARNIVAL'):
            files = os.listdir(exp + '/Results_CARNIVAL')
            files = np.array(sorted(files))
            files = np.delete(files, [0,1], 0)
            files = files[(files != 'results_CARNIVAL.Rdata') & (files != 'weightedModel_1.txt') & (files != 'nodesAttributes_1.txt')]
            count = dg.count_models(files)
            total += count

print(f'The total number of networks is {total}')

MAX_ATOMS = 79
MAX_DEGREE = 21
BOND_FEATURES = 1



