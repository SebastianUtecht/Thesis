from code import InteractiveInterpreter
from Animate import *
from Metrics import *
import os
from sklearn.datasets import make_blobs
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pickle

# data = np.load('data/ubud_mcap_smaller_final.npy', allow_pickle=True)
# p_mask = data[0]
# data = (data[0][p_mask == 1], data[1][p_mask == 1], data[2][p_mask == 1], data[3][p_mask == 1])

# x_range = np.array([-15,15])
# y_range = np.array([-25,20])
# z_range = np.array([-30,32])


# p_mask = data[0]
# x = data[1]
# p = data[2]
# q = data[3]

# data1 = np.load('data/test_sphere_smaller/data.npy', allow_pickle=True)
# data1 = (data1[0][-1]*3, data1[1][-1], data1[2][-1], data1[3][-1])

# p_mask_sphere = data1[0]
# x_sphere = data1[1]
# x_sphere[:,2] -= 3
# x_sphere[:,1] -= 4

# x       = np.concatenate((x, x_sphere),axis=0)
# p_mask  = np.concatenate((p_mask, p_mask_sphere),axis=0)
# p       = np.concatenate((p,data1[2]), axis=0)
# q       = np.concatenate((q,data1[3]), axis=0)

# data = np.load('data/playground_data_smaller.npy', allow_pickle=True)
# p_mask, x, _, _ = data

# general_visualize(x, p_mask, sphere=[[[0,-9,13],10]])

# x_range_mask = (x[:,0] > x_range[0]) *(x[:,0] < x_range[1])
# p_mask = p_mask[x_range_mask]s
# x = x[x_range_mask]
# p = p[x_range_mask]
# q = q[x_range_mask]


# y_range_mask = (x[:,1] > y_range[0]) *(x[:,1] < y_range[1]) 
# p_mask = p_mask[y_range_mask]
# x = x[y_range_mask]
# p = p[y_range_mask]
# q = q[y_range_mask]

# z_range_mask = (x[:,2] > z_range[0]) *(x[:,2] < z_range[1]) 
# p_mask = p_mask[z_range_mask]
# x = x[z_range_mask]
# p = p[z_range_mask]
# q = q[z_range_mask]

# data1 = np.load('data/test_sphere/data.npy', allow_pickle=True)
# data1 = (data1[0][-1]*3, data1[1][-1], data1[2][-1], data1[3][-1])

# p_mask_sphere = data1[0]
# x_sphere = data1[1]
# x_sphere[:,2] -= 4
# x_sphere[:,1] -=4s

# x       = np.concatenate((x, x_sphere),axis=0)
# p_mask  = np.concatenate((p_mask, p_mask_sphere),axis=0)
# p       = np.concatenate((p,data1[2]), axis=0)
# q       = np.concatenate((q,data1[3]), axis=0)

# print(len(x))
# print(len(p_mask))
# print(len(p))
# print(len(q))


# combined_data = (p_mask, x, p, q)

# def save(data_tuple, name, output_folder):
#     with open(f'{output_folder}/{name}.npy', 'wb') as f:
#         pickle.dump(data_tuple, f)

# save(combined_data, 'playground_data', 'data')a
# folder = 'data/playground_data.npy'
# data = np.load(folder, allow_pickle=True)
# p_mask, x, _, _ = data
# general_visualize(x, p_mask, sphere=[[[0,-3.5,10],1]])

# data = np.load('data/ubud_mcap_final.npy', allow_pickle=True)
# p_mask, x, _, _ = data
# general_visualize(x, p_mask, sphere=[[ [0,15,5], 24], [[0,-17.5,20], 25]]) # USE THESE SPHERES FOR BC
# general_visualize(x, p_mask, sphere=[[[0,- 3,12.5], 10]])

# folder = f"data/test_sphere_smaller/data.npy"
# interactive_animate(folder)

# folder = f"datWdata.npy"

# folder = f'data/upper_cube_bound_2/data.npy'
# folder = f'data/aug_cube_bound_tests_higher/data.npy'
# animate_in_vivo(folder, extended_idx=True) 

# data = np.load(folder, allow_pickle=True)
# p_mask, x, _, _ = data
# tstep = 500
# general_visualize(x[tstep], p_mask[tstep], sphere=[[[0,5,0],6]])


# conc_lst = np.linspace(0.5, 0.99, 15)
# conc = conc_lst[3]
# print(conc)


# folder = f"data/primed_MPC_4/data.npy"

# interactive_animate(folder, view_particles='polar')

# folder = "data/diff_coef_test_4/diff_coef_2/data.npy"

# interactive_plot(folder, timestep=-1, polar_idx=4)
diff = 100
str = 10
# folder = f"new_data/chemo_gridsearch/diff_{diff}_str{str}/data.npy"
# folder = "data/INF_TEST_DELETE_AFTER/data.npy"
uec_str = 0.2
mpc_str = 0.1

folder = f"new_data/testing_final_3/data.npy"
# folder = f'new_data/UEC_noise_grid/uec{uec_str}_mpc{mpc_str}/data.npy'
# data = np.load(folder, allow_pickle=True)
# folder = f"new_data/testing_loose_bounds_tot/testing_loose_bounds_14/data.npy"
# p_mask, x, _, _ = data
# p_mask = p_mask[1020]
# p_mask = x[1020]
# general_visualize(x, p_mask)

animate_in_vivo(folder, extended_idx=True)  


# data = np.load(folder, allow_pickle=True)
# mask_lst, x_lst, _, _ = data
# x = x_lst[-1]
# p_mask = mask_lst[-1]

# z_mask, polar_adj_arr = adj_lst_sphere(x, p_mask, radius=5)
# idx = fix_polar_adj_arr(polar_adj_arr, p_mask, z_mask)ww

# visualize_neighbors(x, p_mask, idx)