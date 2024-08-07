from code import InteractiveInterpreter
from Animate import *
from Metrics import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pickle
from DataGeneration import *
from KidneyGenInVitroGradVes import *
import ProduceMCap as mcap

def save_data(data, name, folder='data'):
    try:
        os.mkdir(folder)
    except:
        pass

    with open(folder + '/' + name + '.npy', 'wb') as f:
        pickle.dump(data, f)


# data = make_cylinder_cross(n_v=300, n_h=150, n_c=70, v_len=35, h_len=25, radius=7)
# # general_visualize(data[1], data[0])

# sim_dict = {
#     'output_folder'     : f'correct_ubud',
#     'data'              : data,
#     'dt'                : 0.2,
#     'eta'               : 0.0,
#     'yield_steps'       : 100,
#     'yield_every'       : 50,
#     'lambdas'           : [[0.3,0,0,0],
#                            [0.3,0,0,0],
#                            [0,1.,0,0],
#                            [0, 0.5, 0.48, 0.02]],
#     'pre_lambdas'       : [0.3, 0.3, 1, 1],
#     'crit_ves_size'     : 0,
#     "gamma"             : 0.0,
#     "gauss_multiplier"  : 0.0,
#     "warmup_dur"        : 0,
#     "pre_polar_dur"     : 0,
#     'min_ves_time'      : 0,
#     "seethru"           : 0,
#     "vesicle_alpha"     : 0.4,
#     "tube_alpha"        : 0.2,
#     'betas'             : (0.0,0.0),
#     'max_cells'         : 12_000,
#     'prolif_delay'      : 0,
#     'bound_radius'      : None,
#     'abs_s2s3'          : True,
#     'tube_wall_str'     : 0.3,
#     'new_tube_wall'     : False,
#     'init_k'            : 100,
#     'random_seed'       : 42,
#     'device'            : 'cuda',
#     'dtype'             : torch.float,
#     'notes'             : f'correct_ubud',
#     'avg_q'             : False,
#     'polar_initialization'  : False
# } 

# run_simulation(sim_dict)

# data = np.load('data/correct_ubud/data.npy', allow_pickle=True)
# timestep = 30
# data = (data[0][timestep], data[1][timestep], data[2][timestep], data[3][timestep])

# p_mask = data[0]
# p_mask -= 1
# x = data[1]
# x[:,2] -= 27.5
# x[:,1] += 10 

# save(data, 'ubud_new_translated', 'data')

# general_visualize(data[1], data[0], sphere=[[[0, -5, 12],8]])



# data = np.load('data/ubud_new_translated.npy', allow_pickle=True)
# sim_dict = {
#     'output_folder'     : f'correct_ubud_w_mcap',
#     'data'              : data,
#     'dt'                : 0.2,
#     'eta'               : 0.0,
#     'yield_steps'       : 100,
#     'yield_every'       : 50,
#     'lambdas'           : [0.5,0.5,0,0,0],
#     'pre_lambdas'       : [0.3, 0.3, 1, 1],
#     'crit_ves_size'     : 0,
#     "gamma"             : 0.0,
#     "gauss_multiplier"  : 0.0,
#     "warmup_dur"        : 0,
#     "pre_polar_dur"     : 0,
#     'min_ves_time'      : 0,
#     "seethru"           : 0,
#     "vesicle_alpha"     : 0.4,
#     "alpha"             : 0.2,
#     "vesicle_time"      : 0,  
#     'betas'             : (0.0,0.1),
#     'max_cells'         : 1000,
#     'prolif_delay'      : 0,
#     'bound_radius'      : None,
#     'abs_s2s3'          : True,
#     'tube_wall_str'     : 0.3,
#     'new_tube_wall'     : False,
#     'init_k'            : 100,
#     'random_seed'       : 42,
#     'device'            : 'cuda',
#     'dtype'             : torch.float,
#     'notes'             : f'correct_ubud',
#     'avg_q'             : False,
#     'polar_initialization'  : False
# } 

# mcap.run_simulation(sim_dict)



# data = np.load(folder, allow_pickle=True)
# data = (data[0][-1], data[1][-1], data[2][-1], data[3][-1])
# save(data, 'ubud_mcap_smaller_final', 'data')

folder = "data/ubud_mcap_smaller_final.npy"
data = np.load(folder, allow_pickle=True)

general_visualize(data[1], data[0], sphere=[[[0, 3,5],10]])

# interactive_animate(folder) 


