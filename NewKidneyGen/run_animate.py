from Animate import *
from Metrics import *
import os
from sklearn.datasets import make_blobs
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

folder = "data/testing_bounds0/data.npy"
interactive_animate(folder, view_particles='polar')

# data = np.load(folder, allow_pickle=True)
# mask_lst, x_lst, _, _ = data
# x = x_lst[-1]
# p_mask = mask_lst[-1]

# z_mask, polar_adj_arr = adj_lst_sphere(x, p_mask, radius=5)
# idx = fix_polar_adj_arr(polar_adj_arr, p_mask, z_mask)

# visualize_neighbors(x, p_mask, idx)