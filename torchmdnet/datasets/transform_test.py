import numpy as np
import os
import sys
import glob
import shutil
from mendeleev import element


file_num_start = 0
file_num_end = 20
path_read = r'/home/user/Jie/deepmd_run/work_EC_EMC/data/'
path_save = r'/home/user/Yiting/torchmd-net/torchmdnet/datasets/'
dirs = glob.glob(path_read+'work_EC*')
for dir in dirs[file_num_start:file_num_end]:

    coord_data = np.load(dir+r'/set.000/coord.npy')
    if coord_data.ndim == 2:
        coord_data = coord_data.reshape(coord_data.shape[0], int(coord_data.shape[1] / 3), 3)

    energy_data = np.load(dir+r'/set.000/energy.npy')

    force_data = np.load(dir + r'/set.000/force.npy')
    if force_data.ndim == 2:
        force_data = force_data.reshape(force_data.shape[0], int(force_data.shape[1] / 3), 3)

    coord_data = coord_data.astype(np.float32)
    energy_data = energy_data.astype(np.float32)
    force_data = force_data.astype(np.float32)

    atoms_bytes = open(dir + r'/type_map.raw', 'rb').read()
    atoms = str(atoms_bytes,'utf-8').split(' ')

    atoms_idx_bytes = open(dir + r'/type.raw', 'rb').read()
    atoms_idx_str = str(atoms_idx_bytes, 'utf-8').split(' ')
    atoms_idx = np.array([int(x) for x in atoms_idx_str])

    for idx, atom in enumerate(atoms):
        loc = np.where(atoms_idx==idx)
        atoms_idx[loc] = element(atoms[int(idx)]).atomic_number

    atoms_idx = atoms_idx.astype(np.float32)

    np.save(path_save+'coord_{}.npy'.format(dir.split('/')[-1]), coord_data)
    np.save(path_save+'energy_{}.npy'.format(dir.split('/')[-1]), energy_data)
    np.save(path_save+'force_{}.npy'.format(dir.split('/')[-1]), force_data)
    np.save(path_save+'embed_{}.npy'.format(dir.split('/')[-1]), atoms_idx)


