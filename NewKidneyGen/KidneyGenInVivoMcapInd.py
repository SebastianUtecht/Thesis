### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree
import os
import itertools
import pickle
from time import time
import json

### NOTICE: NOT ALL CODE IN THIS FILE IS WRITTED BY ME (Sebastian Utecht)
### The main model model is based on the paper:  "Theoretical tool bridging cell polarities with development of robust morphologies"
### And the version of the model used here is based on the code from the repository: https://github.com/juliusbierk/polar
### The code from here on will contain comments outlining what I have written myself and what I have not.

##### alpha and lambda indexing #####
# 0:    Mesenchyme progenitor cells         (MPC)
# 1:    Ureteric epithelial cells           (UEC)
# 2:    Interstitial cells                  (IC)
# 3:    Intermediate renal epithelial cells (iREC)
# 4:    Renal epithelial cells              (REC)
# 5:    Pit Ureteric epithelial cells       (pUEC)
# 6:    Primed Mesenchyme cells             (prMPC)
# 7:    Formerly primed iREC cells          (pri_iREC)
# 8:    Formerly primed REC cells           (pri_REC)
# 9:    The tip of the ureteric bud         (tipUEC)

# Class which contains the simulation parameters and functions for the simulation
class Simulation:
    def __init__(self, sim_dict):
        # Setting the simulation parameters

        # General stuff
        self.device         = sim_dict['device']            # 'cuda' or 'cpu'
        self.dtype          = sim_dict['dtype']             # datatype
        self.yield_every    = sim_dict['yield_every']       # Number of timesteps between each yield of data
        self.random_seed    = sim_dict['random_seed']       # Random seed for the simulation

        # Main model parameters
        self.k              = sim_dict['init_k']            # initial number of nearest neighbours found in find_potential_neighbours()
        self.true_neighbour_max = sim_dict['init_k']//2     # maximum number of true nearest neighbour a cell has
        self.dt             = sim_dict['dt']                # timestep
        self.sqrt_dt        = np.sqrt(self.dt)              # square root of timestep (calculated here once, as to not do it every timestep)
        self.eta            = sim_dict['eta']               # noise strength
        self.not_polar_eta  = sim_dict['not_polar_eta']     # noise strength for non-polar cells (MPC and IC cells move than their polar counterparts)
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)           # interaction strengths
        self.UEC_alpha, self.vesicle_alpha, self.tube_alpha = sim_dict['alphas']                                # Alpha values for cells (Facilitates the formation of vesicles and tubes)
        self.alphas         = torch.tensor([0, self.UEC_alpha, 0, self.vesicle_alpha, self.vesicle_alpha], device=self.device, dtype=self.dtype) # -- || --
        self.betas          = torch.tensor(sim_dict['betas'], device=self.device, dtype=self.dtype)             # Beta values for cells (Rate of division)
        self.max_cells      = sim_dict['max_cells']         # Maximum number of cells in the simulation. Simulation will break at this point.
        self.prolif_delay   = sim_dict['prolif_delay']      # Delay before cells start proliferating
        self.seethru        = sim_dict['seethru']                                                               # Number of cells a cell can "see through" (A seethru of 1 allows cells to interact w. an additional layer of voronoi neighbors)
        self.seethru_stop   = sim_dict['seethru_stop']                                                          # Timestep when seethru is  is dialed to 0 (cells no longer have filopodia)
        self.abs_s2s3       = sim_dict['abs_s2s3']          # Boolean which decides if the S2 and S3 terms in the potential should be absolute values
        self.UEC_update_strength = sim_dict['UEC_update_strength']  # Strength of the UEC update (Almost alway 0, but kept as a variable for future use)

        # Attraction parameters
        self.UEC_REC_str    = sim_dict['UEC_REC_str']       # Strength of the UEC-REC/iREC interaction. Overwrites the polar interaction for a non-polar one.           
        self.l3iREC_MPC_min_atr  = sim_dict['l3iREC_MPC_min_atr']                                         # Boolean which decides if iREC and MPC cells with no polarity should interact less with l3 activated REC/iREC cells              
        self.pUEC_REC_str    = sim_dict['pUEC_REC_str']     # Strength of the pUEC-REC/iREC interaction. Overwrites the polar interaction for a non-polar one. Stronger than UEC_REC_str
        self.l3_pUEC_REC_str= torch.tensor(sim_dict['l3_pUEC_REC_str'], device=self.device, dtype=self.dtype)   # l0 governing pUEC-REC/iREC interaction    
        self.tube_wall_str = sim_dict['tube_wall_str']                          # Used to set interaction between tube walls to be non-polar (otherwise all tube walls fuse)
        self.l3_l1_adhesion_str = sim_dict['l3_l1_adhesion_str']                # Strength of the l3-l1 adhesion between iREC/REC cells in order to prevent too much mixing
        self.MPC_distance       = sim_dict['MPC_distance']                      # Distance at which MPC cells start to interact with other cells
        
        #THESE ARE NOT USED IN THE CURRENT VERSION OF THE MODEL
        self.MPC_iREC_l0    = torch.tensor(sim_dict['MPC_iREC_l0'], device=self.device, dtype=self.dtype)       # l0 governing MPC-REC/iREC interaction
        self.UEC_l0    = torch.tensor(sim_dict['UEC_l0'], device=self.device, dtype=self.dtype)                 # l0 governing MPC-UEC interaction
        
        # WNT-MPC cell activation and recruitment parameters
        activation_deats = sim_dict['activation_deats']
        self.activation_type = activation_deats[0]                                                              
        self.activation_loc = torch.tensor(activation_deats[1], device=self.device, dtype=self.dtype)
        if self.activation_type == 'sphere':
            self.activation_radius = activation_deats[2]
        elif self.activation_type == 'cube':
            self.activation_cube_loc = activation_deats[2]
            self.activation_cube_side = activation_deats[3]
        else:
            raise ValueError('Activation type not recognized') 
        
        self.iREC_evo, self.iREC_alpha_evo , self.iREC_evo_tot_time, self.iREC_times  = self.make_iREC_evolution(sim_dict['iREC_time_arr']) # Evolution of the iREC cells. (After activation of MPCs)
        self.polar_time = torch.sum(self.iREC_times[:2]) + 1            # Time when the cells start to polarize
        self.i_ves_time_start = self.polar_time                         # Time when the vesicle formation starts
        self.i_ves_time_end   = torch.sum(self.iREC_times[:3]) + 1      # Time when the vesicle formation ends
        self.recruitment_stop = sim_dict['recruitment_stop']            # Recruitment stops once a set nunber of MPC cells have become REC cells
        self.recruitment_delay = sim_dict['recruitment_delay']          # Delay before recruitment starts
        self.iREC_idx  = torch.tensor([],dtype=torch.int, device=self.device)   # Indeces of iREC cells (used for the evolution of the iREC cells)
        self.iREC_timing = torch.tensor([],dtype=torch.int, device=self.device) # Timing of the iREC cells (used for the evolution of the iREC cells)
        self.REC_IC_ratio   = sim_dict['REC_IC_ratio']      # Ratio of REC to IC cells when for recruitment of MPC (Might be deprecated) 
        self.primed_neigh_threshold  = sim_dict['primed_neigh_threshold']       # Threshold for the number of only ABP cells a primed MPC cell can have as neighbours before it continues in its evolution
        self.ABP_assim_threshold     = sim_dict['ABP_assim_threshold']          # Number of iREC cells recruited before recruitment changes to facilitate new recruits attaching to the bottom of the forming nephron

        # 'Armpit' parameters 
        self.pit_loc = torch.tensor(sim_dict['pit_sphere'][0], device=self.device, dtype=self.dtype)                    # Cells in the "armpit" of the ureteric bud where adhesion is stronger
        self.pit_radius = sim_dict['pit_sphere'][1]                                                                     # Radius of the "armpit" sphere

        # Chemoattractive paramaters for tip UEC
        self.tip_loc = torch.tensor(sim_dict['tip_sphere'][0], device=self.device, dtype=self.dtype)    # Location of the tip UEC cells
        self.tip_radius = sim_dict['tip_sphere'][1]                                                     # Radius of the tip UEC cells
        self.UEC_chemo_diff_coef = sim_dict['UEC_chemo_diff_coef']                                      # Diffusion coefficient of the chemoattractant
        self.UEC_chemo_str = sim_dict['UEC_chemo_str']                                                  # Strength of the chemoattractant


        # iREC/REC bound parameters
        self.bound_type     = sim_dict['bound_type']
        if sim_dict['bound_deats']:
            bound_deats = sim_dict['bound_deats']
            if self.bound_type == 'sphere':
                if len(bound_deats) == 5:
                    self.two_sphere_bc = True
                    self.bound1_loc      = torch.tensor(bound_deats[0], device=self.device, dtype=self.dtype)
                    self.bound1_radius   = bound_deats[1]
                    self.bound2_loc      = torch.tensor(bound_deats[2], device=self.device, dtype=self.dtype)
                    self.bound2_radius   = bound_deats[3]
                    self.bound_str      = bound_deats[4]
                else:
                    self.two_sphere_bc = False
                    self.bound1_loc      = torch.tensor(bound_deats[0], device=self.device, dtype=self.dtype)
                    self.bound1_radius   = bound_deats[1]
                    self.bound_str      = bound_deats[2]
            elif self.bound_type == 'matrigel':
                self.bound1_loc = torch.tensor(bound_deats[0], device=self.device, dtype=self.dtype)
                self.bound_str = bound_deats[1]
            elif self.bound_type == 'cube':
                self.bound_loc  = torch.tensor(bound_deats[0], device=self.device, dtype=self.dtype)
                self.bound_ext  = torch.tensor(bound_deats[1], device=self.device, dtype=self.dtype)
                self.bound_str  = bound_deats[2]
        else:
            self.bound_str = False
        
        # Additional potential parameters parameters
        self.diff_coef      = sim_dict['diff_coef']         # Diffusion coefficient for WNT gradient
        self.WNT_str        = sim_dict['WNT_str']           # Strength of WNT gradient
        self.WNT_c          = sim_dict['WNT_c']             # Distance at which the WNT gradient will not influence direction of tubulogenisis 
        self.avg_q          = sim_dict['avg_q']             # Boolean which decides if the average q should be used in the potential. Most often False
        self.gamma          = sim_dict['gamma']             # Strength of the potential initially making the cells form a vesicle
        self.chemo_atr_N    = sim_dict['chemo_atr_N']       # Number of iREC cells that exude a chemoattractant
        self.chemo_diff_coef= sim_dict['chemo_diff_coef']   # Diffusion coefficient of the chemoattractant
        self.chemo_atr_str  = sim_dict['chemo_atr_str']     # Strength of the chemoattractant
        
        # Stuff that is is initalized empty
        self.beta_tensor = None                 # Tensor for the beta values of the cells. Initialized in the init_cells() function       
        self.d = None                           # Tensor for the distances between cells. Initialized in the potential() function
        self.idx = None                         # Tensor for the indeces of the nearest neighbours. Initialized in the potential() function
        self.assimilation = False               # Bool for if the assimilation has begun
        self.recruitment_stopped = False        # Bool for if the recruitment has stopped

        # Setting the random seed of the simuation
        torch.manual_seed(self.random_seed)     # Setting the random seed for the simulation (Torch)

    #Function for deleting indices from a tensor
    def torch_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool, device=self.device)
        mask[indices] = False
        return tensor[mask]
    
    #Function for making the tensors outlining the evolution of the iREC cells
    def make_iREC_evolution(self, time_arr):
        #number of timesteps for each phase
        i_pref_time     = int(time_arr[0] / self.dt)
        pref_time       = int(time_arr[1] / self.dt)
        i_ves_time      = int(time_arr[2] / self.dt)
        ves_time        = int(time_arr[3] /self.dt)
        i_tube_time     = int(time_arr[4] /self.dt)
        tot_time        = int((np.array(time_arr) / 0.2).sum())

        # Lambda values for the non-polar part of the evolution from MPC to iREC/REC
        non_polar_ipref = torch.linspace(self.lambdas[0][0], 1, i_pref_time, device=self.device)
        non_polar_pref  = torch.ones(pref_time, device=self.device)
        non_polar_ives= torch.linspace(1,0,i_ves_time, device=self.device)
        non_polar_rest  = torch.zeros(ves_time + i_tube_time, device=self.device)
        non_polar_part  = torch.cat((non_polar_ipref, non_polar_pref, non_polar_ives, non_polar_rest), axis=0)
        non_polar_part = non_polar_part.to(self.device)

        # Lambda values for the polar part of the evolution from MPC to iREC/REC
        polar_end_vals  = self.lambdas[4][1:]
        polar_ipref     = torch.zeros((i_pref_time + pref_time,3), device=self.device)
        polar_ives_part1= torch.linspace(0,1, i_ves_time, device=self.device)
        polar_ives      = torch.zeros((i_ves_time, 3), device=self.device)
        polar_ives[:,0] = polar_ives_part1
        polar_ves       = torch.zeros((ves_time, 3), device=self.device)
        polar_ves[:,0]  = torch.ones(ves_time, device=self.device)
        polar_itube     = torch.linspace(0,1, i_tube_time, device=self.device)
        polar_itube     = polar_end_vals[None:,] * polar_itube[:,None]
        polar_itube[:,0]= torch.linspace(1.0,polar_end_vals[0], i_tube_time, device=self.device)
        polar_part      = torch.cat((polar_ipref, polar_ives, polar_ves, polar_itube) , axis=0)
        polar_part = polar_part.to(self.device)

        # Lambda values for the total evolution from MPC to iREC/REC
        total = torch.cat((non_polar_part[:,None],polar_part), axis=1)

        # Alpha values for the total evolution from MPC to iREC/REC
        alpha_pref_ipref= torch.zeros(i_pref_time + pref_time, device=self.device)
        alpha_ives      = torch.ones(i_ves_time, device=self.device) * self.vesicle_alpha  #torch.linspace(0, self.vesicle_alpha, i_ves_time, device=self.device)
        alpha_ves       = torch.ones(ves_time, device=self.device) * self.vesicle_alpha
        alpha_itube     = torch.linspace(self.vesicle_alpha, self.tube_alpha, i_tube_time, device=self.device)
        alpha_total     = torch.cat((alpha_pref_ipref, alpha_ives, alpha_ves, alpha_itube))
        
        return total, alpha_total, tot_time, torch.tensor([i_pref_time, pref_time, i_ves_time, ves_time, i_tube_time], device=self.device, dtype=torch.int)

    # Funciton for finding k nearest neighbours via cKDTree (Still implemented on CPU only, so kinda slow w. transfer and all)
    # NOT WRITTEN SELF
    @staticmethod
    def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
        return d[:, 1:], idx[:, 1:]

    # Function for finding voronoi neighbours (True neighbours) inthe potential neighbours
    # NOT WRITTEN SELF
    def find_true_neighbours(self, d, dx):
        with torch.no_grad():
            z_masks = []
            i0 = 0
            batch_size = 250
            i1 = batch_size
            while True:
                if i0 >= dx.shape[0]:
                    break

                n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
                n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

                z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= self.seethru
                z_masks.append(z_mask)

                if i1 > dx.shape[0]:
                    break
                i0 = i1
                i1 += batch_size
        z_mask = torch.cat(z_masks, dim=0)
        return z_mask
    
    # Function implementing spherical bounds around a give point and with a given radius
    # The bounds are thought of as elastic stroma that is deformed by the expanding nephron and thereby exude a spring potential on the cells
    def bound(self, x, affected_idx):
        affected_cells = x[affected_idx]

        if self.bound_type == 'matrigel':
            bound_dists = torch.sqrt(torch.sum((self.bound_loc - affected_cells)**2, dim=1))
            max_dist    = torch.max(bound_dists)
            v_add       = (1/2 * self.bound_str * bound_dists**2) / max_dist
            v_add       = torch.nan_to_num(v_add, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.sum(v_add)

        elif self.bound_type == 'sphere':
            if self.two_sphere_bc:
                bound1_dists= torch.sqrt(torch.sum((self.bound1_loc -  affected_cells)**2, dim=1))
                bound2_dists= torch.sqrt(torch.sum((self.bound2_loc -  affected_cells)**2, dim=1))
                v_add1      = torch.where(bound1_dists > self.bound1_radius, 1/2 * self.bound_str * (bound1_dists - self.bound1_radius)**2, 0.0)
                v_add2      = torch.where(bound2_dists > self.bound2_radius, 1/2 * self.bound_str * (bound2_dists - self.bound2_radius)**2, 0.0)
                v_add       = torch.min(v_add1, v_add2)

            else:
                bound1_dists= torch.sqrt(torch.sum((self.bound1_loc -  affected_cells)**2, dim=1))
                v_add       = torch.where(bound1_dists > self.bound1_radius, 1/2 * self.bound_str * (bound1_dists - self.bound1_radius)**2, 0.0)
            
            v_add           = torch.nan_to_num(v_add, nan=0.0, posinf=0.0, neginf=0.0)
            v_add           = torch.sum(v_add)

            return v_add

        elif self.bound_type == 'cube':
            bound_dists = affected_cells - self.bound_loc

            # if not self.augment_cube:
            bound_dists = torch.abs(bound_dists)
            bound_dists = torch.max(bound_dists - self.bound_ext, torch.zeros_like(bound_dists))
            v_add       = (1/2 * self.bound_str * bound_dists**2)
            # else:

            # above_ledge_mask = bound_dists[:,2] > - self.cube_ledge
            # bound_dists = torch.abs(bound_dists)
            # bound_dists[above_ledge_mask] = torch.max(bound_dists[above_ledge_mask] - self.bound_ext, torch.zeros_like(bound_dists[above_ledge_mask]))
            # bound_dists[~above_ledge_mask] = torch.max(bound_dists[~above_ledge_mask] - self.ledge_ext, torch.zeros_like(bound_dists[~above_ledge_mask]))
            # v_add       = (1/2 * self.bound_str * bound_dists**2)
            
            v_add       = torch.nan_to_num(v_add, nan=0.0, posinf=0.0, neginf=0.0)
            v_add       = torch.sum(v_add)
            upper_z_potential = torch.where( affected_cells[:,2] > (self.bound_loc[2]), 1/2 * self.bound_str * (affected_cells[:,2] - self.bound_loc[2])**2, 0.0)
            upper_z_potential = torch.sum(upper_z_potential)

            return v_add + upper_z_potential          
    
    # Potential for initially orienting the apical basal polarity of iREC cells when they initially form 
    # the renal vesicle 
    def gauss_grad(self, d, dx, ives_idx, connection_mask):
        with torch.no_grad():
            gauss           = torch.exp(-(d ** 2) * 0.04)
            zero_gauss      = torch.zeros_like(gauss, device=self.device, dtype=self.dtype)
            zero_gauss[ives_idx] = gauss[ives_idx]
            zero_gauss          *= connection_mask
            grad            = torch.sum((zero_gauss[:, :, None] * dx * d[:,:,None]), dim=1)
        return grad
    
    # Potential that aligns the planar cell polarity of cells near the source of the WNT perpendicular to the WNT gradient.
    def WNT_grad(self, x, dx, idx, z_mask, tube_idx):
        with torch.no_grad():
            tube_x, tube_dx, tube_idx_idx = x[tube_idx], dx[tube_idx], idx[tube_idx]
            tube_neigh_pos  = x[tube_idx_idx]
            tube_z          = z_mask[tube_idx]

            WNT_x_dists     = torch.sqrt(torch.sum((self.activation_loc - tube_x)**2, dim=1))
            WNT_neigh_dists = torch.sqrt(torch.sum((self.activation_loc[None,None].expand(tube_neigh_pos.shape) - tube_neigh_pos)**2, dim=2))
            WNT_x           = torch.exp( -WNT_x_dists / self.diff_coef)
            WNT_neigh       = torch.exp( -WNT_neigh_dists / self.diff_coef)

            WNT_grad_  = (WNT_x[:,None] - WNT_neigh) * tube_z
            
            tot_WNT_grad = torch.sum((WNT_grad_)[:,:,None].expand(WNT_neigh.shape[0], WNT_neigh.shape[1],3) * tube_dx, dim=1)
            tot_WNT_grad[torch.sum(tot_WNT_grad, dim=1) != 0.0] /= torch.sqrt(torch.sum(tot_WNT_grad[torch.sum(tot_WNT_grad, dim=1)  != 0.0] ** 2, dim=1))[:, None]
            tot_WNT_grad = torch.nan_to_num(tot_WNT_grad, nan=0.0, posinf=0.0, neginf=0.0)

        return tot_WNT_grad, WNT_x_dists
    
    def chemo_atr_pot(self, x, attractors_mask, attractees_mask):
        #iREC cells furthest away from the WNT source exude a chemoattractant that pulls the cells towards the source
        
        # Attractees positions
        attractees_pos = x[attractees_mask]

        # Origin of the chemoattractive potential
        with torch.no_grad():
            attractors_pos   = x[attractors_mask]
            WNT_x_dists = torch.sqrt(torch.sum((self.activation_loc - attractors_pos)**2, dim=1))
            _, chemo_atr_idx= torch.sort(WNT_x_dists, descending=True)
            chemo_atr_idx   = chemo_atr_idx[:self.chemo_atr_N]                  #Maybe just use 5 attractors
            chemo_atr_pos   = torch.mean(attractors_pos[chemo_atr_idx], dim=0)

        # Potential for the chemoattractant
        chemo_dists     = torch.sqrt(torch.sum((chemo_atr_pos - attractees_pos)**2, dim=1))
        chemo_dists     = chemo_dists[chemo_dists > 6.0]
        chemo_pot       = torch.exp( -chemo_dists / self.chemo_diff_coef)
        chemo_pot = torch.nan_to_num(chemo_pot, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.sum(chemo_pot)
    
    def UEC_chemo_atr_pot(self, x, p_mask_true):
        # UEC cells are attracted to the tip of the ureteric bud by a chemoattractant

        MPC_pos         = x[p_mask_true == 0]

        with torch.no_grad():
            chemo_UEC_pos   = x[p_mask_true == 9]

        min_dists       = torch.sqrt(torch.sum((MPC_pos[None,:] - chemo_UEC_pos[:,None])**2, dim=2)).min(dim=0)[0]
        chemo_pot       = torch.exp( -min_dists / self.UEC_chemo_diff_coef)
        chemo_pot = torch.nan_to_num(chemo_pot, nan=0.0, posinf=0.0, neginf=0.0)
    
        return torch.sum(chemo_pot)

    # Main potential function
    def potential(self, x, p, q, p_mask, l, a, idx, d, tstep):
        # Find neighbours
        # NOT WRITTEN SELF
        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        z_mask = self.find_true_neighbours(d, dx)

        # Minimize size of z_mask and reorder idx and dx
        # NOT WRITTEN SELF
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)
        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(idx, 1, sort_idx)
        m = torch.max(torch.sum(z_mask, dim=1)) + 1
        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]

        p_mask_true         = p_mask.clone() # We keep the true p_mask
        p_mask              = p_mask.clone() # Used so we don't overwrite the original p_mask
        p_mask[p_mask == 9] = 1              # For almost all intents and purposes, the tipUEC cells are treated as UEC cells

        # If the simulation is in the assimilation phase, the cells are updated to the new cell types
        if self.assimilation:
            p_mask[p_mask == 6] = 0     # Primed MPC cells mostly treated as MPC
            p_mask[p_mask == 7] = 3     # iREC after assimilation has begun
            p_mask[p_mask == 8] = 4     # REC after assimilation has begun    
            true_interaction_mask = torch.cat((p_mask_true[:,None].expand(p_mask_true.shape[0], idx.shape[1])[:,:,None], p_mask_true[idx][:,:,None]), dim=2)

            # Masks for facilitating: - No strong pUEC-iREC/REC interaction for cells activated after assimilation
            #                         - Possible stronger interactions between l1 activated iREC and primed MPC

            pUEC_new_iREC_mask = torch.sum(true_interaction_mask == torch.tensor([5,7], device=self.device), dim=2) == 2
            pUEC_new_iREC_mask = torch.logical_or( torch.sum(true_interaction_mask == torch.tensor([7,5], device=self.device), dim=2) == 2, pUEC_new_iREC_mask)

            pUEC_new_REC_mask = torch.sum(true_interaction_mask == torch.tensor([5,8], device=self.device), dim=2) == 2
            pUEC_new_REC_mask = torch.logical_or( torch.sum(true_interaction_mask == torch.tensor([8,5], device=self.device), dim=2) == 2, pUEC_new_REC_mask)
            pUEC_new_RECiREC_mask = torch.logical_or(pUEC_new_iREC_mask, pUEC_new_REC_mask) # Facilitating no strong pUEC interaction for late recruits

        # Mask that contains information about what type of cells are interacting
        interaction_mask = torch.cat((p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])[:,:,None], p_mask[idx][:,:,None]), dim=2)

        # Masks for different types of interactions
        # This looks a bit excessive, but there are 8 different types of cells and therefore ultimately
        # 36 different types of interactions. This is the most efficient way I could think of to implement this.

        MPC_tot_mask = torch.any(interaction_mask == 0, dim=2)

        MPC_iREC_mask = torch.sum(interaction_mask == torch.tensor([0,3], device=self.device), dim=2) == 2
        MPC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,0], device=self.device), dim=2) == 2, MPC_iREC_mask)
        
        MPC_UEC_mask = torch.sum(interaction_mask == torch.tensor([0,1], device=self.device), dim=2) == 2
        MPC_UEC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([1,0], device=self.device), dim=2) == 2, MPC_UEC_mask)

        pUEC_UEC_mask = torch.sum(interaction_mask == torch.tensor([5,1], device=self.device), dim=2) == 2
        pUEC_UEC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([1,5], device=self.device), dim=2) == 2, pUEC_UEC_mask)

        UEC_iREC_mask = torch.sum(interaction_mask == torch.tensor([1,3], device=self.device), dim=2) == 2
        UEC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,1], device=self.device), dim=2) == 2, UEC_iREC_mask)
        UEC_REC_mask = torch.sum(interaction_mask == torch.tensor([1,4], device=self.device), dim=2) == 2
        UEC_REC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([4,1], device=self.device), dim=2) == 2, UEC_REC_mask)
        UEC_RECiREC_mask = torch.logical_or(UEC_iREC_mask, UEC_REC_mask)

        pUEC_iREC_mask = torch.sum(interaction_mask == torch.tensor([5,3], device=self.device), dim=2) == 2
        pUEC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,5], device=self.device), dim=2) == 2, pUEC_iREC_mask)
        pUEC_REC_mask = torch.sum(interaction_mask == torch.tensor([5,4], device=self.device), dim=2) == 2
        pUEC_REC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([4,5], device=self.device), dim=2) == 2, pUEC_REC_mask)
        pUEC_RECiREC_mask = torch.logical_or(pUEC_iREC_mask, pUEC_REC_mask)

        iREC_iREC_mask       = torch.sum(interaction_mask == torch.tensor([3,3], device=self.device), dim=2) == 2
        REC_iREC_mask        = torch.sum(interaction_mask == torch.tensor([3,4], device=self.device), dim=2) == 2
        REC_iREC_mask        = torch.logical_or( torch.sum(interaction_mask == torch.tensor([4,3], device=self.device), dim=2) == 2, REC_iREC_mask)
        RECiREC_RECiREC_mask = torch.logical_or(iREC_iREC_mask, REC_iREC_mask)

        REC_REC_mask = torch.sum(interaction_mask == torch.tensor([4,4], device=self.device), dim=2) == 2
        allREC_mask  = torch.logical_or(RECiREC_RECiREC_mask, REC_REC_mask) 

        # Normalise dx
        # NOT WRITTEN SELF
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        l_i     = l[:, None, :].expand(p.shape[0], idx.shape[1], -1)
        l_j     = l[idx, :]
        l_min   = torch.min(l_i, l_j)

        # Setting interaction between MPC iREC's such that the MPC's 'chase' after the iREC --> Cell migration into forming nephron
        l_min[MPC_iREC_mask] = torch.tensor([self.MPC_iREC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)

        # Slightly altering the interaction between pUEC and UEC cells so they don't mix
        # Currently the UEC are update-locked, but the line is kept for now
        # l_min[pUEC_UEC_mask]  *= 0.9

        if torch.sum(allREC_mask) > 0:
            # This section sets it so, that every interaction between iREC particles have a relaxation distance of 2
            # Except for the very early iREC recruits which still have increasing l0.
            add_arr1 = torch.zeros_like(l_min[RECiREC_RECiREC_mask], device=self.device)
            add_arr1[:,0] = (1 - torch.sum(l_min[RECiREC_RECiREC_mask], dim=1))
            add_arr2      = torch.roll(add_arr1, 1, dims=1)
            mask1 = l_i[RECiREC_RECiREC_mask][:,0] > 0.0
            mask2 = l_j[RECiREC_RECiREC_mask][:,0] > 0.0
            mask = torch.logical_or(mask1, mask2)
            add_arr = torch.where(mask[:,None], add_arr1, add_arr2)
            l_min_clone = l_min.clone()
            l_min[RECiREC_RECiREC_mask] += add_arr

            mask1 = ~torch.any(l_i[RECiREC_RECiREC_mask][:,1:], dim=1)
            mask2 = ~torch.any(l_j[RECiREC_RECiREC_mask][:,1:], dim=1)
            mask  = torch.logical_or(mask1, mask2)
            l_min[RECiREC_RECiREC_mask] = torch.where(mask[:,None], l_min_clone[RECiREC_RECiREC_mask], l_min[RECiREC_RECiREC_mask])

            # Weakening the interaction between PCP polarized iREC/REC cells and non PCP polarized iREC/REC cells
            # very slightly such that these subpopulations don't mix as much
            pcp_vals_i  = l_i[RECiREC_RECiREC_mask][:,3] > 0.0
            pcp_vals_j  = l_j[RECiREC_RECiREC_mask][:,3] > 0.0
            pcp_vals    = torch.logical_xor(pcp_vals_i, pcp_vals_j)
            l_min[RECiREC_RECiREC_mask] = torch.where(pcp_vals[:,None], l_min[RECiREC_RECiREC_mask] * self.l3_l1_adhesion_str, l_min_clone[RECiREC_RECiREC_mask])

        # Setting interactions between non-polar cells and cells with some degree of activated polarity (full vesicle or star l3)
        # so they don't interact as much. We want new recruits to skip the already l3 activated cells to integrate with newer recruits.
        if self.l3iREC_MPC_min_atr:

            # THIS FACILITATES LOW ATTACHMENT TO FORMING VESICLES.  
            # DONT THINK NECESSARY BUT KEEPING IT COMMENTED OUT FOR NOW
            # if self.low_attach_ives:
            #     mask1 = torch.logical_or(l_i[:,:,3] > 0.0, l_i[:,:,1] == 1.0)
            #     mask2 = ~torch.any(l_j[:,:,1:], dim=2)
            #     mask4 = ~torch.any(l_i[:,:,1:], dim=2)
            #     mask5 = torch.logical_or(l_j[:,:,3] > 0.0, l_j[:,:,1] == 1.0)

            mask1 = l_i[:,:,3] > 0.0
            mask2 = ~torch.any(l_j[:,:,1:], dim=2)
            mask3 = torch.logical_and(mask1, mask2)

            mask4 = ~torch.any(l_i[:,:,1:], dim=2)
            mask5 = l_j[:,:,3] > 0.0
            mask6 = torch.logical_and(mask4, mask5)

            mask = torch.logical_or(mask3, mask6)

            # Setting interaction between l3 activated iREC and MPC to same as
            # UEC and MPC. UEC is also l3 activated, so this makes sense in that way.
            # Code also overwrites l_min[MPC_UEC_mask] = torch.tensor([self.UEC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)
            l_min[mask] = torch.tensor([self.UEC_l0, 0, 0, 0], device=self.device, dtype=self.dtype)

        # Setting the interaction between pUEC and l3 activated iREC and REC higher to facilitate attachment of the tube.
        mask = l_min[pUEC_RECiREC_mask][:,3] > 0.0 
        pUEC_REC_tensor = torch.tensor([self.pUEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)
        l3_pUEC_REC_tensor = torch.tensor([self.l3_pUEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)
        l_min[pUEC_RECiREC_mask] = torch.where(mask[:,None] ,l3_pUEC_REC_tensor, pUEC_REC_tensor)

        if self.assimilation:
            # New cell recruits do not flock to the pUEC, but to the bottom of the structure.
            l_min[pUEC_new_RECiREC_mask] = torch.tensor([self.UEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)

        # Setting the interaction between UEC and iREC and REC cells to be non-polar
        l_min[UEC_RECiREC_mask]  = torch.tensor([self.UEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)
        
        #Finding the alpha values used for the interactions (minimum value of the two interacting cells)
        # NOT WRITTEN SELF
        a_i = a[:, None].expand(p.shape[0], idx.shape[1])
        a_j = a[idx]
        a_min = torch.min(a_i, a_j)[:,:,None].expand(a_i.shape[0], a_i.shape[1], 3)

        # Setting the polarities such that the final potential can be calculated 
        # In one big cross product
        # NOT WRITTEN SELF
        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        # Use avg. q or not? The result seems to be almost the same, but the avg. q is used in the paper
        if self.avg_q:
            avg_q = (qi + qj)*0.5
        else:
            avg_q = qi

        # Mask of cells with PCP polarity
        aniso_mask = l_min[:,:,3] > 0.0

        # Code here facilitates wedging between cells with PCP polarity
        ts = (avg_q * dx).sum(axis = 2)
        angle_dx = avg_q * ts[:,:,None]

        angle_dx[~aniso_mask] = dx[~aniso_mask]

        if self.seethru != 0:
            angle_dx[aniso_mask] = angle_dx[aniso_mask] * d[:,:,None][aniso_mask]

        # NOT WRITTEN SELF
        pi_tilde = pi - a_min * angle_dx
        pj_tilde = pj + a_min * angle_dx
    
        # Calculating the potential
        polar_mask0 = torch.sum(l_i[:,:,1:], dim=2) > 0.0
        polar_mask1 = torch.sum(l_j[:,:,1:], dim=2) > 0.0
        interactions_w_polarity = torch.logical_and(polar_mask0, polar_mask1)

        pi_tilde[interactions_w_polarity] = pi_tilde[interactions_w_polarity]/torch.sqrt(torch.sum(pi_tilde[interactions_w_polarity] ** 2, dim=1))[:, None]               # The p-tildes are normalized
        pj_tilde[interactions_w_polarity] = pj_tilde[interactions_w_polarity]/torch.sqrt(torch.sum(pj_tilde[interactions_w_polarity] ** 2, dim=1))[:, None]
        
        # Doing cross products for the potential
        # NOT WRITTEN SELF
        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)            # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

        # Setting the S2 and S3 terms to be absolute values (nematic PCP instead of vector)
        if self.abs_s2s3:
            S2 = torch.abs(S2)
            S3 = torch.abs(S3)

        # Setting interaction between cells, which are part of different part of the structure
        # Were this not here, every tube would fuse, and the tightly packed S-shape would be lost 
        if self.tube_wall_str:
            with torch.no_grad():
                wall_mask = (torch.sum(pi * pj , dim = 2) < 0.0) * (torch.sum(-dx * pj , dim = 2) < 0.0)    
                wall_mask = torch.logical_and(wall_mask , allREC_mask)
                l_min[wall_mask] = torch.tensor([self.tube_wall_str, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        # NOT WRITTEN SELF
        S = l_min[:,:,0] + l_min[:,:,1] * S1 + l_min[:,:,2] * S2 + l_min[:,:,3] * S3
        Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
        Vij[MPC_tot_mask] = torch.where(d[MPC_tot_mask] > self.MPC_distance , 0.0, Vij[MPC_tot_mask])
        Vij = torch.nan_to_num(Vij, nan=0.0, posinf=0.0, neginf=0.0)
        Vij_sum = torch.sum(Vij)

        # Applying the gamma potential for the vesicle formation
        ives_idx = self.iREC_idx[(self.iREC_timing > self.i_ves_time_start) * (self.iREC_timing < self.i_ves_time_end)]
        if self.gamma > 0.0 and len(ives_idx) > 0:
            gauss_grad = self.gauss_grad(d, dx, ives_idx, RECiREC_RECiREC_mask)
            Vi       = torch.sum(self.gamma * p * gauss_grad, dim=1)
            Vi       = torch.nan_to_num(Vi, nan=0.0, posinf=0.0, neginf=0.0)
            Vij_sum -= torch.sum(Vi)
        
        # Applying the WNT-gradient potential for the tube formation
        tube_idx = torch.argwhere(l[:,-1] > 0.0)[:,0]
        if self.WNT_str > 0.0 and len(tube_idx) > 0:
            WNT_grad_, WNT_x_dists   = self.WNT_grad(x=x, dx=dx, idx=idx, z_mask=z_mask, tube_idx=tube_idx)
            S4          = (1.0 - torch.sum(q[tube_idx] * WNT_grad_, dim=1)**2)
            cells_affected = WNT_x_dists < self.WNT_c

            # Can be commented in, if you want the potential to be as written in the paper
            # This allows cells to move such that they minimize their potential with regards to the WNT gradient
            # instead of just being turned such that their PCP is perpendicular to the WNT gradient
            # S4         *= torch.exp(-d[tube_idx]/5)
            # S4          = torch.sum(S4[:,None] * z_mask[tube_idx], dim=1) / torch.sum(z_mask[tube_idx], dim=1)
            
            Vij_sum    -= self.WNT_str * torch.sum(cells_affected * S4)
        
        # Applying the chemoattractant potential for the proximal recruitment of iREC cells
        if self.assimilation and self.chemo_atr_str > 0.0:
            polarity_mask = torch.any(l[:,1:], dim=1)
            attractors_mask =  torch.logical_or( (p_mask == 3) , (p_mask == 4) ) * polarity_mask                                               #(p_mask == 3) * l[:,3] == 0.0 * torch.any(l[:,1:], dim=1) 
            attractees_mask = torch.logical_or( (p_mask_true == 6), ( (p_mask == 3) * (~polarity_mask) ) )

            if torch.sum(attractees_mask)  > 0 and torch.sum(attractors_mask) > 0:
                chemo_pot = self.chemo_atr_pot(x=x, attractors_mask = attractors_mask, attractees_mask = attractees_mask)
                Vij_sum -= self.chemo_atr_str * chemo_pot
        
        # Applying the chemoattractant potentialfor the the MPC cells to be attracted to the tip of the ureteric bud
        if self.UEC_chemo_str > 0.0:
            UEC_chemo_pot = self.UEC_chemo_atr_pot(x=x, p_mask_true=p_mask_true)
            Vij_sum -= self.UEC_chemo_str * UEC_chemo_pot
        
        # If bounds are present we apply the potential
        if self.bound_str > 0.0:
            affected_idx = torch.argwhere(torch.any(l[:,1:], dim=1))[:,0]
            bc      =  self.bound(x=x, affected_idx = affected_idx)
            Vij_sum += bc

        return Vij_sum, int(m), idx

    # Function for initiating the simulation
    def init_simulation(self, x, p, q, p_mask):
        #Check that everything is good shape wise
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

        # Make tensors and transfer to right device with right datatype
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        l = torch.zeros((x.shape[0],4) , dtype=self.dtype, device=self.device)
        a = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
        self.beta_tensor   = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)

        # Lambda values are set
        for i in range(len(self.lambdas)):
            l[p_mask == i] = self.lambdas[i]
            a[p_mask == i] = self.alphas[i]
            self.beta_tensor[p_mask == i] = self.betas[i]

        #pUEC cells are found (pit ureteric epithelial cells)
        pUEC = torch.sum(( self.pit_loc -  x)**2, dim=1) < self.pit_radius**2
        p_mask[torch.logical_and(pUEC, p_mask == 1)] = 5

        #tipUEC cells are found (tip ureteric epithelial cells)
        tipUEC = torch.sum(( self.tip_loc -  x)**2, dim=1) < self.tip_radius**2
        p_mask[torch.logical_and(tipUEC, p_mask == 1)] = 9

        return x, p, q, p_mask, l, a 

    #Updating number of found potential neighbours
    # NOT WRITTEN SELF
    def update_k(self, true_neighbour_max):
        k = self.k
        fraction = true_neighbour_max / k                                                         # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                                                                       # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:                                                                     # Vice versa
            k = int(1.5 * k)
        self.k = k                                                                                # We update k
        return k
    
    # Function which returns a bool of whether or not to update the neighbours
    def update_neighbors_bool(self, tstep, division):
        if division == True:
            return True
        n_update = 1 if tstep < 100 else max([1, int(20 * np.tanh(tstep / 200))])
        return ((tstep % n_update) == 0)

    # Function that progresses the simulation one time step
    def time_step(self, x, p, q, p_mask, l, a, tstep):

        # Update seethru if it is to stop
        if tstep == (self. seethru_stop + 1):
            self.seethru = 0

        # Start with cell division
        division, x, p, q, p_mask, l, a, self.beta_tensor = self.cell_division(x=x, p=p, q=q, p_mask=p_mask, l=l, a=a, tstep=tstep)

        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        # NOT WRITTEN SELF
        k = self.update_k(self.true_neighbour_max)
        k = min(k, len(x) - 1)

        # We find potential neighbours
        # PARTLY WRITTEN SELF
        if self.update_neighbors_bool(tstep, division):
            d, idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
        idx = self.idx
        d = self.d

        # Normalise p, q
        with torch.no_grad():
            cells_w_polarity = torch.sum(l[:,1:], dim=1) > 0.0
            p[cells_w_polarity] /= torch.sqrt(torch.sum(p[cells_w_polarity] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
            q[cells_w_polarity] /= torch.sqrt(torch.sum(q[cells_w_polarity] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

            p[~cells_w_polarity] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)
            q[~cells_w_polarity] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)

        # Calculate potential
        V, self.true_neighbour_max, idx = self.potential(x, p, q, p_mask, l, a, idx, d, tstep=tstep)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            # The gradients are found
            x_grads = x.grad
            p_grads = p.grad
            q_grads = q.grad

            #Different cell types are updated differently
            #UEC is almost never updated, as we treat it statically
            #MPC is updated with a large eta to facilitate greater random movement
            
            # Masks for the different cell types are found
            UEC_mask = torch.logical_or(torch.logical_or((p_mask == 1), (p_mask == 5)), (p_mask == 9))
            not_UEC_mask  = ~UEC_mask
            no_polar_mask = ~torch.any(l[:,1:], dim=1)
            no_polar_not_UEC_mask   = torch.logical_and(not_UEC_mask, no_polar_mask)
            not_UEC_mask[no_polar_not_UEC_mask]  = 0  

            #We update the polar cells which are not UEC (iRE/REC)
            x[not_UEC_mask] += -x_grads[not_UEC_mask] * self.dt + self.eta * torch.empty(*x[not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            p[not_UEC_mask] += -p_grads[not_UEC_mask] * self.dt + self.eta * torch.empty(*x[not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q[not_UEC_mask] += -q_grads[not_UEC_mask] * self.dt + self.eta * torch.empty(*x[not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
           
           #We update the non-polar cells which are not UEC (MPC)
            x[no_polar_not_UEC_mask] += -x_grads[no_polar_not_UEC_mask] * self.dt + self.not_polar_eta * torch.empty(*x[no_polar_not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            p[no_polar_not_UEC_mask] += -p_grads[no_polar_not_UEC_mask] * self.dt + self.not_polar_eta * torch.empty(*x[no_polar_not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q[no_polar_not_UEC_mask] += -q_grads[no_polar_not_UEC_mask] * self.dt + self.not_polar_eta * torch.empty(*x[no_polar_not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            #We update the UEC cells
            x[UEC_mask] += self.UEC_update_strength * (-x_grads[UEC_mask] * self.dt + self.eta * torch.empty(*x[UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
            p[UEC_mask] += self.UEC_update_strength * (-p_grads[UEC_mask] * self.dt + self.eta * torch.empty(*x[UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
            q[UEC_mask] += self.UEC_update_strength * (-q_grads[UEC_mask] * self.dt + self.eta * torch.empty(*x[UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)

            #Gradiants are zero'd out
            p.grad.zero_()
            q.grad.zero_()
            x.grad.zero_()

        #If any iREC cells are present, we check whether they should evolve to REC cells
        if len(self.iREC_timing) > 0:
            self.iREC_timing += 1                                                                   #Incrementing the timing of the iREC cells        
            fin_REC_bool = self.iREC_timing >= self.iREC_evo_tot_time                               #Checking if any iREC cells should evolve to REC cells
            if torch.any(fin_REC_bool):
                fin_REC_idx = self.iREC_idx[fin_REC_bool]                                           #Finding the indices of the iREC cells that should evolve to REC cells
                p_mask[fin_REC_idx] = torch.where(p_mask[fin_REC_idx] == 3, 4, 8).type(torch.int)   #Changing the cell type to REC
                removal_idx = torch.argwhere(torch.isin(self.iREC_idx, fin_REC_idx))[:,0]           #Delete the indices of the new REC cells off the iREC lists
                self.iREC_idx = self.torch_delete(self.iREC_idx, removal_idx)                       #Delete the indices of the new REC cells off the iREC lists
                self.iREC_timing = self.torch_delete(self.iREC_timing, removal_idx)                 #Delete the timing of the new REC cells off the iREC lists
                l[fin_REC_idx] = self.lambdas[4]                                                    #Setting the lambda and alpha values of the new REC cells
                a[fin_REC_idx] = self.tube_alpha
                self.beta_tensor[fin_REC_idx] = self.betas[4]                                       #Setting the beta values of the new REC cells
            l[self.iREC_idx] = self.iREC_evo[self.iREC_timing]                                      #Updating the lambda values of the iREC cells
            a[self.iREC_idx] = self.iREC_alpha_evo[self.iREC_timing]                                #Updating the alpha values of the iREC cells            

        # Check for new types
        # If we havent reached the recruitment stop, we check for new recruits
        start_recruitment = tstep > self.recruitment_delay
        end_recruitment = (torch.sum(p_mask == 4) 
                        + torch.sum(p_mask == 8)
                        + torch.sum(p_mask == 3)
                        + torch.sum(p_mask == 7))  >= self.recruitment_stop
        
        if end_recruitment and not self.recruitment_stopped:
            p_mask[p_mask == 6] = 0
            
            self.recruitment_stopped = True

        if start_recruitment and not end_recruitment:
            
            # Finding MPC distances to the WNT source and monte carlo sample them according
            # To diffusion in order to find changing cells 
            WNT_dists = torch.sqrt(torch.sum((self.activation_loc -  x)**2, dim=1))

            if self.activation_type == "sphere":
                cells_within_act = WNT_dists < self.activation_radius
            elif self.activation_type == "cube":
                cells_within_act = torch.logical_and(torch.logical_and(torch.abs(self.activation_cube_loc[0] - x[:,0]) < self.activation_cube_side[0], 
                                                                      torch.abs(self.activation_cube_loc[1] - x[:,1]) < self.activation_cube_side[1]), 
                                                                      torch.abs(self.activation_cube_loc[2] - x[:,2]) < self.activation_cube_side[2])
                cells_within_act *= x[:,2] < self.activation_cube_loc[2]
            else:
                raise ValueError("Activation type not recognized")
            
            changing_cells   = cells_within_act * (p_mask == 0)
            changing_cells  *= (torch.exp(-WNT_dists / self.diff_coef) 
                                > (torch.rand_like(changing_cells, dtype=self.dtype, device=self.device)) / self.dt )
            
            # If vesicle is already formed then newly activated cells should only attach to the 
            # bottom of the structure. This is facilitated by introducing primed MPC cells that only
            # start evolving when close to a threshold neighbors of not l3 activated iREC cells.
            if self.assimilation:
                primed_cells_idx = torch.argwhere(changing_cells)[:,0]
                p_mask[primed_cells_idx] = 6
                
                primed_cell_neighbors   = idx[p_mask == 6]
                iREC_mask               = torch.logical_or((p_mask[primed_cell_neighbors] == 3), (p_mask[primed_cell_neighbors] == 7))

                l1_vals = l[:,1][primed_cell_neighbors]
                l3_vals = l[:,3][primed_cell_neighbors]
                num_ves_neighbors = torch.sum( (l1_vals > 0.0) * (l3_vals == 0.0) * iREC_mask, dim=1)
                primed_changing_cells = num_ves_neighbors >= self.primed_neigh_threshold
                
                changing_cells_idx = torch.argwhere(p_mask == 6)[:,0][primed_changing_cells]

            else:
                changing_cells_idx = torch.argwhere(changing_cells)[:,0]
                if (torch.sum(p_mask == 3) + torch.sum(p_mask == 4) ) > self.ABP_assim_threshold: #Before: torch.sum(l[p_mask == 3][:,1] > 0)
                    self.assimilation = True

            if not(torch.any(changing_cells)):
                return x, p, q, p_mask, l, a
            else: 
                REC_IC_mask = torch.rand(len(changing_cells_idx), device=self.device) < self.REC_IC_ratio
                iREC_cells = changing_cells_idx[REC_IC_mask]
                IC_cells = changing_cells_idx[~REC_IC_mask]

                
                # If assimilation has begun the iREC cells behave slightly differently
                if self.assimilation:
                    p_mask[iREC_cells] = 7
                else:
                    p_mask[iREC_cells] = 3
                
                self.beta_tensor[iREC_cells] = self.betas[3]

                p_mask[IC_cells] = 2

                if not(torch.any(self.iREC_idx)):
                    self.iREC_idx   = iREC_cells.clone()
                    self.iREC_timing= torch.zeros_like(iREC_cells)
                else:
                    self.iREC_idx   = torch.cat((self.iREC_idx,iREC_cells), dim=0)
                    self.iREC_timing= torch.cat((self.iREC_timing,torch.zeros_like(iREC_cells)), dim=0)

                l[IC_cells] = self.lambdas[2]
                a[IC_cells] = self.alphas[2]

        return x, p, q, p_mask, l, a
    
    # Function runs the simulation
    # PARTLY WRITTEN SELF
    def simulation(self, x, p, q, p_mask):
        x, p, q, p_mask, l, a = self.init_simulation(x, p, q, p_mask)

        tstep = 0
        while True:
            tstep += 1
            x, p, q, p_mask, l, a = self.time_step(x, p, q, p_mask, l, a, tstep)

            if tstep % self.yield_every == 0 or len(x) > self.max_cells:
                xx = x.detach().to("cpu").numpy().copy()
                pp = p.detach().to("cpu").numpy().copy()
                qq = q.detach().to("cpu").numpy().copy()
                ll = l.detach().to("cpu").numpy().copy()
                aa = a.detach().to("cpu").numpy().copy()
                pp_mask = p_mask.detach().to("cpu").numpy().copy()
                yield xx, pp, qq, pp_mask, ll, aa
    
    # Function for finding positions for new polar cells in division
    # The new cells are placed orthogonal to the ABP of the dividing cells
    # such that they stay in the epithelial layer
    def get_prolif_positions(self, p, q, p_mask, mask_ind):
        """
        Gives moves orthogonal to ABP of the dividing cells
        """
        with torch.no_grad():
            # We find all the polar particles
            p_polar = p[p_mask == mask_ind].clone().squeeze()
            q_polar = q[p_mask == mask_ind].clone().squeeze()

            # Special case if we only have 1 polar particle proliferating
            if torch.numel(p_polar) == 3:
                # Gram-Schmidt orthanormalization
                p_polar /= torch.sqrt(torch.sum(p_polar ** 2))
                q_polar -= torch.sum(q_polar * p_polar) * p_polar
                q_polar /= torch.sqrt(torch.sum(q_polar ** 2))

                # Matrix for linear transformation
                lin_trans = torch.zeros((3, 3), device=self.device, dtype=self.dtype)
                lin_trans[:,0] = p_polar
                lin_trans[:,1] = q_polar
                lin_trans[:,2] = torch.cross(p_polar,q_polar)

                # Find move in transformed space and transform back
                new_pos =  torch.zeros_like(p_polar, device=self.device).squeeze()
                new_pos[1:] = torch.normal(mean = 0.0, std = 1.0, size=(1,2)).squeeze()
                new_pos = (lin_trans @ new_pos).squeeze()
            else:
                # Gram-Schmidt orthanormalization
                p_polar /= torch.sqrt(torch.sum(p_polar ** 2, dim=1))[:, None]
                q_polar -= torch.sum(q_polar * p_polar, dim=1)[:,None] * p_polar
                q_polar /= torch.sqrt(torch.sum(q_polar ** 2, dim=1))[:, None]

                # Matrix for linear transformation
                lin_trans = torch.zeros((len(p_polar), 3, 3), device=self.device, dtype=self.dtype)
                lin_trans[:,:,0] = p_polar
                lin_trans[:,:,1] = q_polar
                lin_trans[:,:,2] = torch.cross(p_polar,q_polar, dim=1)

                # Find move in transformed space and transform back
                new_pos =  torch.zeros_like(p_polar)
                new_pos[:,1:] = torch.normal(mean = 0.0, std = 1.0, size=(p_polar.shape[0],2)).squeeze()
                new_pos = (lin_trans @ new_pos[:,:,None]).squeeze()

            return new_pos
    
    #Function for cell division
    # PARTLY WRITTEN SELF
    def cell_division(self, x, p, q, p_mask, l, a, tstep, beta_decay = 1.0):

        beta = self.beta_tensor
        dt = self.dt

        if torch.sum(beta) < 1e-5 or tstep <= self.prolif_delay:
            return False, x, p, q, p_mask, l, a, beta

        # set probability according to beta and dt
        d_prob = beta * dt
        # flip coins
        draw = torch.empty_like(beta).uniform_()
        # find successes
        events = draw < d_prob
        division = False

        if torch.sum(events) > 0:
            with torch.no_grad():
                division = True
                # find cells that will divide
                idx = torch.nonzero(events)[:, 0]
                
                x0      = x[idx, :]
                p0      = p[idx, :]
                q0      = q[idx, :]
                l0      = l[idx, :]
                a0      = a[idx]
                p_mask0 = p_mask[idx]
                b0      = beta[idx] * beta_decay
                
                new_iREC_idx    = torch.argwhere(torch.isin(idx, self.iREC_idx))[:,0]
                if torch.any(new_iREC_idx):
                    new_iREC_idx        = len(x) + new_iREC_idx - 1
                    self.iREC_idx       = torch.cat((self.iREC_idx, new_iREC_idx), dim=0)
                    dividing_iREC_idx   = self.iREC_idx[torch.isin(self.iREC_idx, idx)]
                    self.iREC_timing    = torch.cat((self.iREC_timing, dividing_iREC_idx), dim=0)

                # make a random vector and normalize to get a random direction
                move = torch.empty_like(x0).normal_()
                move[p_mask0 == 1] = self.get_prolif_positions(p0, q0, p_mask0, mask_ind=1)
                move[p_mask0 == 4] = self.get_prolif_positions(p0, q0, p_mask0, mask_ind=4)
                move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

                # place new cells
                x0 = x0 + move

                # append new cell data to the system state
                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                l = torch.cat((l, l0))
                a = torch.cat((a, a0))
                p_mask = torch.cat((p_mask, p_mask0))
                beta = torch.cat((beta, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, p_mask, l, a, beta

#Function for saving data 
# NOT WRITTEN SELF
def save(data_tuple, name, output_folder):
    with open(f'{output_folder}/{name}.npy', 'wb') as f:
        pickle.dump(data_tuple, f)

#Function for running simulation
# PARTLY WRITTEN SELF
def run_simulation(sim_dict):
    # Make the simulation runner object:
    data_tuple = sim_dict.pop('data')
    verbose    = sim_dict.pop('verbose')
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every']
    data_yield = sim_dict['data_yield'] 

    np.random.seed(sim_dict['random_seed'])
    p_mask, x, p, q = data_tuple
        
    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)

    output_folder = sim_dict['output_folder']

    try:
        os.mkdir('data')
    except:
        pass
    try: 
        os.mkdir('data/' + output_folder)
    except:
        pass

    x_lst = []
    p_lst = []
    q_lst = []
    p_mask_lst = []
    l_lst = []
    a_lst = []

    with open(f'data/' + output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    notes = sim_dict['notes']

    if not notes:
        notes = output_folder

    if verbose:
        print('Starting simulation with notes:')
        print(notes)

    i = 0
    t1 = time()

    for xx, pp, qq, pp_mask, ll, aa in itertools.islice(runner, yield_steps):
        i += 1
        if verbose:
            print(f'Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end='\r')

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        p_mask_lst.append(pp_mask)
        l_lst.append(ll)
        a_lst.append(aa)
        
        if len(xx) > sim_dict['max_cells']:
            break

        if i % data_yield == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst, l_lst, a_lst), name='data', output_folder='data/' + output_folder)

    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst, l_lst, a_lst), name='data', output_folder='data/' + output_folder)