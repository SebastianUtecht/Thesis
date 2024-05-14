from code import InteractiveInterpreter
from Animate import *
from Metrics import *
import os
from sklearn.datasets import make_blobs
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# folder = "data/wedging_gridsearch/size4000_conc0.9_beta0.0001/data.npy" 

# folder = "data/testing_comma2/data.npy"
folder = "data/low_nonpolar_adh/data.npy"
# interactive_animate(folder, view_particles='polar')
# interactive_plot(folder,timestep=-1, view_particles='polar')


folder = "data/invivo_gradtube_longer/data.npy"
animate_in_vivo(folder, view_particles=None)  
# animate_in_vivo(folder)


# data = np.load(folder, allow_pickle=True)
# mask_lst, x_lst, _, _ = data
# x = x_lst[-1]
# p_mask = mask_lst[-1]

# z_mask, polar_adj_arr = adj_lst_sphere(x, p_mask, radius=5)
# idx = fix_polar_adj_arr(polar_adj_arr, p_mask, z_mask)

# visualize_neighbors(x, p_mask, idx)