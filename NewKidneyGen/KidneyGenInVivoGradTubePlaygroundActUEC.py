### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree
import os
import itertools
import gc
import pickle
from time import time
import json
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)

#### alpha and lambda indexing ####
# 0:    Mesenchyme progenitor cells         (MPC)
# 1:    Ureteric epithelial cells           (UEC)
# 2:    Interstitial cells                  (IC)
# 3:    Intermediate renal epithelial cells (iREC)
# 4:    Renal epithelial cells              (REC)
# 5:    Pit Ureteric epithelial cells       (pUEC)
 
class Simulation:
    def __init__(self, sim_dict):
        self.device         = sim_dict['device']
        self.dtype          = sim_dict['dtype']
        self.k              = sim_dict['init_k'] 
        self.true_neighbour_max = sim_dict['init_k']//2
        self.dt             = sim_dict['dt']
        self.sqrt_dt        = np.sqrt(self.dt)
        self.eta            = sim_dict['eta']
        self.not_polar_eta  = sim_dict['not_polar_eta']
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)
        self.UEC_l0    = torch.tensor(sim_dict['UEC_l0'], device=self.device, dtype=self.dtype)
        self.MPC_iREC_l0    = torch.tensor(sim_dict['MPC_iREC_l0'], device=self.device, dtype=self.dtype)
        self.l3_pUEC_REC_str= torch.tensor(sim_dict['l3_pUEC_REC_str'], device=self.device, dtype=self.dtype)
        self.iREC_nopolar_attr_str  = sim_dict['iREC_nopolar_attr_str']
        self.low_attach_ives = sim_dict['low_attach_ives']
        self.seethru        = sim_dict['seethru']
        self.seethru_stop   = sim_dict['seethru_stop']
        self.UEC_alpha, self.vesicle_alpha, self.tube_alpha = sim_dict['alphas'] 
        self.alphas         = torch.tensor([0, self.UEC_alpha, 0, self.vesicle_alpha, self.tube_alpha], device=self.device, dtype=self.dtype)
        self.betas          = torch.tensor(sim_dict['betas'], device=self.device, dtype=self.dtype)
        self.activation_loc = torch.tensor(sim_dict['activation_sphere'][0], device=self.device, dtype=self.dtype)
        self.activation_radius = sim_dict['activation_sphere'][1]
        self.pit_loc = torch.tensor(sim_dict['pit_sphere'][0], device=self.device, dtype=self.dtype)
        self.pit_radius = sim_dict['pit_sphere'][1]

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

        self.REC_IC_ratio   = sim_dict['REC_IC_ratio']
        self.max_cells      = sim_dict['max_cells']
        self.abs_s2s3       = sim_dict['abs_s2s3']
        self.diff_coef      = sim_dict['diff_coef']
        self.WNT_str        = sim_dict['WNT_str']
        self.WNT_c          = sim_dict['WNT_c']
        self.UEC_update_strength = sim_dict['UEC_update_strength']
        self.UEC_REC_str    = sim_dict['UEC_REC_str']
        self.pUEC_REC_str    = sim_dict['pUEC_REC_str']
        self.yield_every    = sim_dict['yield_every']
        self.random_seed    = sim_dict['random_seed']
        self.vesicle_time   = sim_dict['vesicle_time']
        self.avg_q          = sim_dict['avg_q']
        self.gamma          = sim_dict['gamma']
        self.iREC_evo, self.iREC_alpha_evo , self.iREC_evo_tot_time, self.iREC_times  = self.make_iREC_evolution(sim_dict['iREC_time_arr'])
        self.polar_time = torch.sum(self.iREC_times[:2]) + 1
        self.i_ves_time_start = self.polar_time
        self.i_ves_time_end   = torch.sum(self.iREC_times[:3]) + 1
        self.iREC_idx  = torch.tensor([],dtype=torch.int, device=self.device)
        self.iREC_timing = torch.tensor([],dtype=torch.int, device=self.device)
        self.tube_wall_str = sim_dict['tube_wall_str']

        ### For playground purposes ###
        self.num_partitions     = sim_dict['num_partitions']
        self.partition_delay    = sim_dict['partition_delay']
        self.partition_mask     = None
        self.partition_counter  = 0
        self.recruitment_sim    = sim_dict['recruitment_sim']
        self.recruit_finished   = sim_dict['recruit_finished']        
        if self.recruitment_sim:
            self.recruitment_loc     = torch.tensor(self.recruitment_sim[0], device=self.device, dtype=self.dtype)
            self.recruitment_radius  = self.recruitment_sim[1]
        self.recruitment_start = sim_dict['recruitment_start']
        self.recruitment_stop = sim_dict['recruitment_stop'] 
        self.augment_cube = sim_dict['augment_cube']
        if self.augment_cube:
            self.cube_ledge = self.augment_cube[0]
            self.ledge_width= self.augment_cube[1]
            self.ledge_ext = self.bound_ext + torch.tensor([0.0,self.ledge_width,0.0], device=self.device, dtype=self.dtype)
        self.recruitment_num = sim_dict['recruitment_num']
        self.update_pUEC = False

        self.sim_time   = 0
        self.vesicle_formation = False
        self.tube_formation    = False
        self.beta_tensor = None 
        self.d = None
        self.idx = None
        self.attachment_cells_found = False

        torch.manual_seed(self.random_seed)

    def torch_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool, device=self.device)
        mask[indices] = False
        return tensor[mask]
    
    def make_iREC_evolution(self, time_arr):
        #number of timesteps for each phase
        i_pref_time     = int(time_arr[0] / self.dt)
        pref_time       = int(time_arr[1] / self.dt)
        i_ves_time      = int(time_arr[2] / self.dt)
        ves_time        = int(time_arr[3] /self.dt)
        i_tube_time     = int(time_arr[4] /self.dt)
        tot_time        = int((np.array(time_arr) / 0.2).sum())

        non_polar_ipref = torch.linspace(self.lambdas[0][0], 1, i_pref_time, device=self.device)
        non_polar_pref  = torch.ones(pref_time, device=self.device)
        non_polar_ives= torch.linspace(1,0,i_ves_time, device=self.device)
        non_polar_rest  = torch.zeros(ves_time + i_tube_time, device=self.device)
        non_polar_part  = torch.cat((non_polar_ipref, non_polar_pref, non_polar_ives, non_polar_rest), axis=0)
        non_polar_part = non_polar_part.to(self.device)

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

        total = torch.cat((non_polar_part[:,None],polar_part), axis=1)

        alpha_pref_ipref= torch.zeros(i_pref_time + pref_time, device=self.device)
        alpha_ives      = torch.ones(i_ves_time, device=self.device) * self.vesicle_alpha  #torch.linspace(0, self.vesicle_alpha, i_ves_time, device=self.device)
        alpha_ves       = torch.ones(ves_time, device=self.device) * self.vesicle_alpha
        alpha_itube     = torch.linspace(self.vesicle_alpha, self.tube_alpha, i_tube_time, device=self.device)
        alpha_total     = torch.cat((alpha_pref_ipref, alpha_ives, alpha_ves, alpha_itube))
        
        return total, alpha_total, tot_time, torch.tensor([i_pref_time, pref_time, i_ves_time, ves_time, i_tube_time], device=self.device, dtype=torch.int)

    @staticmethod
    def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):
        tree = cKDTree(x)
        d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
        return d[:, 1:], idx[:, 1:]

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
    
    def bound(self, x):

        if self.bound_type == 'matrigel':
            bound_dists = torch.sqrt(torch.sum((self.bound_loc - x)**2, dim=1))
            max_dist    = torch.max(bound_dists)
            v_add       = (1/2 * self.bound_str * bound_dists**2) / max_dist
            return torch.sum(v_add)

        elif self.bound_type == 'sphere':
            if self.two_sphere_bc:
                bound1_dists= torch.sqrt(torch.sum((self.bound1_loc -  x)**2, dim=1))
                bound2_dists= torch.sqrt(torch.sum((self.bound2_loc -  x)**2, dim=1))
                v_add1      = torch.where(bound1_dists > self.bound1_radius, 1/2 * self.bound_str * (bound1_dists - self.bound1_radius)**2, 0.0)
                v_add2      = torch.where(bound2_dists > self.bound2_radius, 1/2 * self.bound_str * (bound2_dists - self.bound2_radius)**2, 0.0)
                v_add       = torch.min(v_add1, v_add2)

            else:
                bound1_dists= torch.sqrt(torch.sum((self.bound1_loc -  x)**2, dim=1))
                v_add       = torch.where(bound1_dists > self.bound1_radius, 1/2 * self.bound_str * (bound1_dists - self.bound1_radius)**2, 0.0)

            v_add           = torch.sum(v_add)

            return v_add

        elif self.bound_type == 'cube':
            bound_dists = x - self.bound_loc
            if not self.augment_cube:
                bound_dists = torch.abs(bound_dists)
                bound_dists = torch.max(bound_dists - self.bound_ext, torch.zeros_like(bound_dists))
                v_add       = (1/2 * self.bound_str * bound_dists**2)
            else:
                above_ledge_mask = bound_dists[:,2] > - self.cube_ledge
                bound_dists = torch.abs(bound_dists)
                bound_dists[above_ledge_mask] = torch.max(bound_dists[above_ledge_mask] - self.bound_ext, torch.zeros_like(bound_dists[above_ledge_mask]))
                bound_dists[~above_ledge_mask] = torch.max(bound_dists[~above_ledge_mask] - self.ledge_ext, torch.zeros_like(bound_dists[~above_ledge_mask]))
                v_add       = (1/2 * self.bound_str * bound_dists**2)
            
            v_add       = torch.sum(v_add)
            upper_z_potential = torch.where( x[:,2] > (self.bound_loc[2]), 1/2 * self.bound_str * (x[:,2] - self.bound_loc[2])**2, 0.0)
            upper_z_potential = torch.sum(upper_z_potential)

            return v_add + upper_z_potential          

    
    def gauss_grad(self, d, dx, ives_idx, connection_mask):
        with torch.no_grad():
            gauss           = torch.exp(-(d ** 2) * 0.04)
            zero_gauss      = torch.zeros_like(gauss, device=self.device, dtype=self.dtype)
            zero_gauss[ives_idx] = gauss[ives_idx]
            zero_gauss          *= connection_mask
            grad            = torch.sum((zero_gauss[:, :, None] * dx * d[:,:,None]), dim=1)
        return grad
    
    def WNT_grad(self, x, dx, idx, z_mask, tube_idx):
        with torch.no_grad():
            tube_x, tube_dx, tube_idx_idx = x[tube_idx], dx[tube_idx], idx[tube_idx]
            tube_neigh_pos  = x[tube_idx_idx]
            tube_z          = z_mask[tube_idx]

            WNT_x_dists     = torch.sqrt(torch.sum((self.activation_loc - tube_x)**2, dim=1))
            WNT_neigh_dists = torch.sqrt(torch.sum((self.activation_loc[None,None].expand(tube_neigh_pos.shape) - tube_neigh_pos)**2, dim=2))
            WNT_x           = torch.exp( -WNT_x_dists / self.diff_coef)
            WNT_neigh       = torch.exp( -WNT_neigh_dists / self.diff_coef)
            
            WNT_grad  = (WNT_x[:,None] - WNT_neigh) * tube_z
            tot_WNT_grad = torch.sum((WNT_grad)[:,:,None].expand(WNT_neigh.shape[0], WNT_neigh.shape[1],3) * tube_dx, dim=1)
            tot_WNT_grad[torch.sum(tot_WNT_grad, dim=1) != 0.0] /= torch.sqrt(torch.sum(tot_WNT_grad[torch.sum(tot_WNT_grad, dim=1)  != 0.0] ** 2, dim=1))[:, None]
        return tot_WNT_grad, WNT_x_dists

    # def WNT_grad_pUEC(self, x, dx, idx, p_mask,  z_mask, tube_idx):
    #     with torch.no_grad():
    #         tube_x, tube_dx, tube_idx_idx = x[tube_idx], dx[tube_idx], idx[tube_idx]
    #         tube_neigh_pos  = x[tube_idx_idx]
    #         tube_z          = z_mask[tube_idx]

    #         #average distance from tube_x to x[p_mask == 5]
    #         WNT_x_dists     = (x[p_mask == 5] - tube_x)**2
    #         WNT_x_dists     = torch.sqrt(torch.sum((self.activation_loc - tube_x)**2, dim=1))
            
    
    def potential(self, x, p, q, p_mask, l, a, idx, d, tstep):
        # Find neighbours

        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        z_mask = self.find_true_neighbours(d, dx)

        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)
        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(idx, 1, sort_idx)
        m = torch.max(torch.sum(z_mask, dim=1)) + 1
        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]

        interaction_mask = torch.cat((p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])[:,:,None], p_mask[idx][:,:,None]), dim=2)

        MPC_iREC_mask = torch.sum(interaction_mask == torch.tensor([0,3], device=self.device), dim=2) == 2
        MPC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,0], device=self.device), dim=2) == 2, MPC_iREC_mask)

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
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        l_i     = l[:, None, :].expand(p.shape[0], idx.shape[1], -1)
        l_j     = l[idx, :]
        l_min   = torch.min(l_i, l_j)

        # Setting interaction between MPC and iREC cells such that the MPC 'chase' after the iREC
        l_min[MPC_iREC_mask] = torch.tensor([self.MPC_iREC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)

        # Slightly altering the interaction between pUEC and UEC cells so they don't mix
        # l_min[pUEC_UEC_mask]  *= 0.9

        if torch.sum(RECiREC_RECiREC_mask) > 0:
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

            # Setting interactions between non-polar cells and cells with some degree of activated polarity (full vesicle or start l3)
            #so they don't interact as much. We want new recruits to skip the already l3 activated cells to integrate with newer recruits.
            if self.iREC_nopolar_attr_str:
                if self.low_attach_ives:
                    mask1 = torch.logical_or(l_i[:,:,3] > 0.0, l_i[:,:,1] == 1.0)
                    mask2 = ~torch.any(l_j[:,:,1:], dim=2)
                    mask4 = ~torch.any(l_i[:,:,1:], dim=2)
                    mask5 = torch.logical_or(l_j[:,:,3] > 0.0, l_j[:,:,1] == 1.0)
                else:
                    mask1 = l_i[:,:,3] > 0.0
                    mask2 = ~torch.any(l_j[:,:,1:], dim=2)
                    mask4 = ~torch.any(l_i[:,:,1:], dim=2)
                    mask5 = l_j[:,:,3] > 0.0

                mask3 = torch.logical_and(mask1, mask2)
                mask6 = torch.logical_and(mask4, mask5)
                mask = torch.logical_or(mask3, mask6)

                l_min[mask] = torch.tensor([self.iREC_nopolar_attr_str, 0, 0, 0], device=self.device, dtype=self.dtype)

        # Setting the interaction between pUEC and l3 activated iREC and REC higher to facilitate attachment of the tube.
        # if torch.sum(p_mask == 4) < self.recruitment_stop:

        if not self.update_pUEC:
            mask = l_min[pUEC_RECiREC_mask][:,3] > 0.0
            pUEC_REC_tensor = torch.tensor([self.pUEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)
            l3_pUEC_REC_tensor = torch.tensor([self.l3_pUEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)
            l_min[pUEC_RECiREC_mask] = torch.where(mask[:,None] ,l3_pUEC_REC_tensor, pUEC_REC_tensor)
        else:
            l_min[pUEC_RECiREC_mask] *= 0.95

        # else:
        #     l_min[pUEC_RECiREC_mask] = self.lambdas[1]
        #     a_min                    = self.alphas[4]

        # Setting the attraction between not polarized iREC cells and pUEC to standard UEC-REC values so they don't flock into the start of the tube 
        # mask = ~torch.any(l_min[pUEC_RECiREC_mask][:,1:], dim=1)
        # l_min[pUEC_RECiREC_mask] = torch.where(mask[:,None], torch.tensor([self.UEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype), l_min[pUEC_RECiREC_mask])

        # Setting the interaction between UEC and iREC and REC cells to be non-polar
        l_min[UEC_RECiREC_mask]  = torch.tensor([self.UEC_REC_str, 0, 0, 0], device=self.device, dtype=self.dtype)
        MPC_UEC_mask = torch.sum(interaction_mask == torch.tensor([0,1], device=self.device), dim=2) == 2
        MPC_UEC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([1,0], device=self.device), dim=2) == 2, MPC_UEC_mask)
        l_min[MPC_UEC_mask] = torch.tensor([self.UEC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)
    
        a_i = a[:, None].expand(p.shape[0], idx.shape[1])
        a_j = a[idx]
        a_min = torch.min(a_i, a_j)[:,:,None].expand(a_i.shape[0], a_i.shape[1], 3)

        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        if self.avg_q:
            avg_q = (qi + qj)*0.5
        else:
            avg_q = qi

        aniso_mask = l_min[:,:,3] > 0.0

        ts = (avg_q * dx).sum(axis = 2)
        angle_dx = avg_q * ts[:,:,None]

        angle_dx[~aniso_mask] = dx[~aniso_mask]

        if self.seethru != 0:
            angle_dx[aniso_mask] = angle_dx[aniso_mask] * d[:,:,None][aniso_mask]

        pi_tilde = pi - a_min * angle_dx
        pj_tilde = pj + a_min * angle_dx

        polar_mask0 = torch.sum(l_i[:,:,1:], dim=2) > 0.0
        polar_mask1 = torch.sum(l_j[:,:,1:], dim=2) > 0.0
        interactions_w_polarity = torch.logical_and(polar_mask0, polar_mask1)

        pi_tilde[interactions_w_polarity] = pi_tilde[interactions_w_polarity]/torch.sqrt(torch.sum(pi_tilde[interactions_w_polarity] ** 2, dim=1))[:, None]               # The p-tildes are normalized
        pj_tilde[interactions_w_polarity] = pj_tilde[interactions_w_polarity]/torch.sqrt(torch.sum(pj_tilde[interactions_w_polarity] ** 2, dim=1))[:, None]

        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)            # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

        if self.abs_s2s3:
            S2 = torch.abs(S2)
            S3 = torch.abs(S3)
                    
        if self.tube_wall_str != None:
            with torch.no_grad():
                wall_mask = torch.sum(pi * pj , dim = 2) < 0.0
                wall_mask = torch.logical_and(wall_mask , allREC_mask)
                l_min[wall_mask] = torch.tensor([self.tube_wall_str, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

        S = l_min[:,:,0] + l_min[:,:,1] * S1 + l_min[:,:,2] * S2 + l_min[:,:,3] * S3
        
        Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
        Vij_sum = torch.sum(Vij)

        ives_idx = self.iREC_idx[(self.iREC_timing > self.i_ves_time_start) * (self.iREC_timing < self.i_ves_time_end)]
        if self.gamma > 0.0 and len(ives_idx) > 0:
            gauss_grad = self.gauss_grad(d, dx, ives_idx, RECiREC_RECiREC_mask)
            Vi       = torch.sum(self.gamma * p * gauss_grad, dim=1)
            Vij_sum -= torch.sum(Vi)
        
        tube_idx = torch.argwhere(l[:,-1] > 0.0)[:,0]
        if self.WNT_str > 0.0 and len(tube_idx) > 0:
            WNT_grad, WNT_x_dists   = self.WNT_grad(x=x, dx=dx, idx=idx, z_mask=z_mask, tube_idx=tube_idx)
            S4          = (1.0 - torch.sum(q[tube_idx] * WNT_grad, dim=1)**2)
            cells_affected = WNT_x_dists < self.WNT_c

            #Augmented versions of the potential. Can be commented in if you want to use the potential as written in the paper.
            # S4         *= torch.exp(-d[tube_idx]/5)  #KOMMENTER IND HVIS DU VIL HA' POTENTIALET SOM SKREVET I PAPERET
            # S4          = torch.sum(S4[:,None] * z_mask[tube_idx], dim=1) / torch.sum(z_mask[tube_idx], dim=1)
            
            Vij_sum    -= self.WNT_str * torch.sum(cells_affected * S4)
        
        if self.bound_str > 0.0:
            bc       =  self.bound(x=x)
            Vij_sum += bc

        return Vij_sum, int(m)

    def init_simulation(self, x, p, q, p_mask):
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        l = torch.zeros((x.shape[0],4) , dtype=self.dtype, device=self.device)
        a = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
        self.beta_tensor   = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)

        for i in range(len(self.betas)):
            self.beta_tensor[p_mask == i] = self.betas[i]
        
        sphere_avg_pos = torch.mean(x[p_mask == 3], dim=0)
        x_trans_vec  = sphere_avg_pos - self.activation_loc
        x_trans_vec  /= torch.sqrt(torch.sum(x_trans_vec ** 2))

        def transfer_matrix(x_trans_vec):
            nx, ny, nz = x_trans_vec
            trans_mat = torch.tensor([[nx, -nx*ny/(1+nz), 1-(nx**2/(1+nz))],
                                      [ny, 1-(ny**2/(1+nz)), -nx*ny/(1+nz)],
                                      [nz,        -ny,          -nx]],
                                      device=self.device, dtype=self.dtype)
            return trans_mat
        
        trans_mat = transfer_matrix(x_trans_vec)
        inv_trans_mat = torch.linalg.inv(trans_mat)
        expanded_inv_trans_mat = inv_trans_mat[None,:,:].repeat_interleave(len(x),axis=0)
        x_trans   = (expanded_inv_trans_mat @ x[:,:,None]).squeeze()
        
        vesicle_x_trans = x_trans[p_mask == 3]

        part_lines = torch.linspace(torch.min(vesicle_x_trans[:,0]).item()-0.1, torch.max(vesicle_x_trans[:,0]).item()+0.1, self.num_partitions + 1, device=self.device, dtype=self.dtype)
        self.partition_mask = torch.ones(len(x_trans), device=self.device, dtype=torch.int) * -1
        for i in range(self.num_partitions):
            cells_in_partition = torch.argwhere((x_trans[:,0] > part_lines[i]) * (x_trans[:,0] < part_lines[i+1]) * (p_mask == 3))[:,0]

            self.partition_mask[cells_in_partition] = i
        
        self.partition_times = torch.arange(0,self.num_partitions) * self.partition_delay

        for i in range(len(self.lambdas)):
            l[p_mask == i] = self.lambdas[i]
            a[p_mask == i] = self.alphas[i]

        pUEC = torch.sum(( self.pit_loc -  x)**2, dim=1) < self.pit_radius**2
        p_mask[torch.logical_and(pUEC, p_mask == 1)] = 5

        return x, p, q, p_mask, l, a 

    def update_k(self, true_neighbour_max):
        k = self.k
        fraction = true_neighbour_max / k                                                         # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                                                                       # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:                                                                     # Vice versa
            k = int(1.5 * k)
        self.k = k                                                                                # We update k
        return k
    
    def update_neighbors_bool(self, tstep, division):
        if division == True:
            return True
        n_update = 1 if tstep < 100 else max([1, int(20 * np.tanh(tstep / 200))])
        return ((tstep % n_update) == 0)

    def time_step(self, x, p, q, p_mask, l, a, tstep):

        if torch.sum(p_mask == 4) >= self.recruitment_start:
            self.update_pUEC = True

        self.sim_time += self.dt

        if tstep == (self. seethru_stop + 1):
            self.seethru = 0

        # Start with cell division
        division, x, p, q, p_mask, l, a, self.beta_tensor = self.cell_division(x=x, p=p, q=q, p_mask=p_mask, l=l, a=a)

        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        k = self.update_k(self.true_neighbour_max)
        k = min(k, len(x) - 1)

        if self.update_neighbors_bool(tstep, division):
            d, idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
        idx = self.idx
        d = self.d

        # Normalise p, q
        with torch.no_grad():
            cells_w_p = torch.any(l[:,1:], dim=1)
            cells_w_q = torch.any(l[:,2:], dim=1)
  
            p[cells_w_p] /= torch.sqrt(torch.sum(p[cells_w_p] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
            q[cells_w_q] /= torch.sqrt(torch.sum(q[cells_w_q] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

            p[~cells_w_p] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)
            q[~cells_w_q] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)

        # Calculate potential
        V, self.true_neighbour_max = self.potential(x, p, q, p_mask, l, a, idx, d, tstep=tstep)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x_grads = x.grad
            p_grads = p.grad
            q_grads = q.grad

            # THIS IS USED FOR MAKING pUEC UPDATE WHEN WE STOP RECRUITMENT
            # COMMENTED OUT FOR NOW
            # if torch.sum(p_mask == 4) < self.recruitment_stop:
            #   UEC_mask = torch.logical_or((p_mask == 1), (p_mask == 5))
            # else:
            #   UEC_mask = (p_mask == 1)

            # Comment this out if you comment the above in
            UEC_mask    = (p_mask == 1)
            pUEC_mask   = (p_mask == 5)
            tot_UEC_mask = torch.logical_or(UEC_mask, pUEC_mask)

            not_UEC_mask  = ~tot_UEC_mask
            no_polar_mask = ~torch.any(l[:,1:], dim=1)
            no_polar_not_UEC_mask   = torch.logical_and(not_UEC_mask, no_polar_mask)
            not_UEC_mask[no_polar_not_UEC_mask]  = 0  

            x[not_UEC_mask] += -x_grads[not_UEC_mask] * self.dt + self.eta * torch.empty(*x[not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            p[not_UEC_mask] += -p_grads[not_UEC_mask] * self.dt + self.eta * torch.empty(*x[not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q[not_UEC_mask] += -q_grads[not_UEC_mask] * self.dt + self.eta * torch.empty(*x[not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
           
            x[no_polar_not_UEC_mask] += -x_grads[no_polar_not_UEC_mask] * self.dt + self.not_polar_eta * torch.empty(*x[no_polar_not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            p[no_polar_not_UEC_mask] += -p_grads[no_polar_not_UEC_mask] * self.dt + self.not_polar_eta * torch.empty(*x[no_polar_not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q[no_polar_not_UEC_mask] += -q_grads[no_polar_not_UEC_mask] * self.dt + self.not_polar_eta * torch.empty(*x[no_polar_not_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            if not self.update_pUEC:
                x[tot_UEC_mask] += self.UEC_update_strength * (-x_grads[tot_UEC_mask] * self.dt + self.eta * torch.empty(*x[tot_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
                p[tot_UEC_mask] += self.UEC_update_strength * (-p_grads[tot_UEC_mask] * self.dt + self.eta * torch.empty(*x[tot_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
                q[tot_UEC_mask] += self.UEC_update_strength * (-q_grads[tot_UEC_mask] * self.dt + self.eta * torch.empty(*x[tot_UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
            else:
                x[UEC_mask] += self.UEC_update_strength * (-x_grads[UEC_mask] * self.dt + self.eta * torch.empty(*x[UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
                p[UEC_mask] += self.UEC_update_strength * (-p_grads[UEC_mask] * self.dt + self.eta * torch.empty(*x[UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
                q[UEC_mask] += self.UEC_update_strength * (-q_grads[UEC_mask] * self.dt + self.eta * torch.empty(*x[UEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)

                x[pUEC_mask] +=  (-x_grads[pUEC_mask] * self.dt + self.eta * torch.empty(*x[pUEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
                p[pUEC_mask] +=  (-p_grads[pUEC_mask] * self.dt + self.eta * torch.empty(*x[pUEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)
                q[pUEC_mask] +=  (-q_grads[pUEC_mask] * self.dt + self.eta * torch.empty(*x[pUEC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt)


            p.grad.zero_()
            q.grad.zero_()
            x.grad.zero_()

        if len(self.iREC_timing) > 0:
            self.iREC_timing += 1
            fin_REC_bool = self.iREC_timing >= self.iREC_evo_tot_time
            if torch.any(fin_REC_bool):
                fin_REC_idx = self.iREC_idx[fin_REC_bool]
                p_mask[fin_REC_idx] = 4
                self.beta_tensor[fin_REC_idx] = self.betas[4]
                removal_idx = torch.argwhere(torch.isin(self.iREC_idx, fin_REC_idx))[:,0]
                self.iREC_idx = self.torch_delete(self.iREC_idx, removal_idx)
                self.iREC_timing = self.torch_delete(self.iREC_timing, removal_idx)
                l[fin_REC_idx] = self.lambdas[4]
                a[fin_REC_idx] = self.tube_alpha
            l[self.iREC_idx] = self.iREC_evo[self.iREC_timing]
            a[self.iREC_idx] = self.iREC_alpha_evo[self.iREC_timing]

        #Check for new types
        if self.sim_time <= (self.partition_times[-1] + self.dt):

            if self.sim_time > self.partition_times[self.partition_counter]:
                changing_cells = self.partition_mask == self.partition_counter
                self.partition_counter += 1
            else:
                return x, p, q, p_mask, l, a
            
            changing_cells_idx = torch.argwhere(changing_cells)[:,0]
            REC_IC_mask = torch.rand(len(changing_cells_idx), device=self.device) < self.REC_IC_ratio
            iREC_cells = changing_cells_idx[REC_IC_mask]
            IC_cells = changing_cells_idx[~REC_IC_mask]
            p_mask[iREC_cells] = 3
            p_mask[IC_cells] = 2
            self.beta_tensor[iREC_cells] = self.betas[3]
            self.beta_tensor[IC_cells] = self.betas[2]

            if not(torch.any(self.iREC_idx)):
                self.iREC_idx   = iREC_cells.clone()
                self.iREC_timing= torch.zeros_like(iREC_cells)
            else:
                self.iREC_idx   = torch.cat((self.iREC_idx,iREC_cells), dim=0)
                self.iREC_timing= torch.cat((self.iREC_timing,torch.zeros_like(iREC_cells)), dim=0)

            l[IC_cells] = self.lambdas[2]
            a[IC_cells] = self.alphas[2]
            # a[self.iREC_idx] = self.alphas[3]

        return x, p, q, p_mask, l, a

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
                pp_mask = p_mask.detach().to("cpu").numpy().copy()
                yield xx, pp, qq, pp_mask
    
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

    def recruitment_sim_func(self, x, p, q, p_mask, l, a):
        
        #Comment in for use of recruitment radius instead of cells furthest away proliferation 
        # dists = torch.sqrt(torch.sum((x - self.recruitment_loc) ** 2, dim=1))
        # cells_in_radius = dists < self.recruitment_radius

        #Comment in for use of cells furthest away proliferation
        dists = torch.sqrt(torch.sum((x - self.activation_loc) ** 2, dim=1))
        _, prolif_idx = torch.topk(dists * (p_mask == 3), self.recruitment_num, largest=True)
        cells_in_radius = torch.zeros(len(x), device=self.device, dtype=torch.int)
        cells_in_radius[prolif_idx] = 1


        if torch.sum(cells_in_radius) == 0:
            return False, x, p, q, p_mask, l, a, self.beta_tensor

        beta_copy = self.beta_tensor.clone()
        beta = self.beta_tensor * cells_in_radius.float()

        if torch.sum(beta) < 1e-5:
            return False, x, p, q, p_mask, l, a, beta

        # set probability according to beta and dt
        d_prob = beta * self.dt
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

                if self.recruit_finished:
                    x0      = x[idx, :]
                    p0      = p[idx, :]
                    q0      = q[idx, :]

                    l0      = self.lambdas[4] * torch.ones_like(l[idx, :])
                    a0      = self.alphas[4] * torch.ones_like(a[idx])
                    p_mask0 = 4 * torch.ones_like(p_mask[idx])
                    b0      = self.betas[4] * torch.ones_like(beta[idx])    

                else:
                    x0      = x[idx, :]
                    p0      = p[idx, :]
                    q0      = q[idx, :]
                    l0      = l[idx, :]
                    a0      = a[idx]
                    p_mask0 = p_mask[idx]
                    b0      = beta[idx]
                    
                    new_iREC_idx    = torch.argwhere(torch.isin(idx, torch.argwhere(p_mask ==3)[:,0]))[:,0]
                    if torch.numel(new_iREC_idx) > 0:
                        new_iREC_idx        = len(x) + new_iREC_idx - 1
                        self.iREC_idx       = torch.cat((self.iREC_idx, new_iREC_idx), dim=0)
                        self.iREC_timing    = torch.cat((self.iREC_timing, torch.zeros_like(new_iREC_idx)), dim=0)

                # make a random vector and normalize to get a random direction
                move = torch.zeros_like(x0)
                move[:,1] = -2

                # place new cells
                x0 = x0 + move

                # append new cell data to the system state
                x = torch.cat((x, x0))
                p = torch.cat((p, p0))
                q = torch.cat((q, q0))
                l = torch.cat((l, l0))
                a = torch.cat((a, a0))
                p_mask = torch.cat((p_mask, p_mask0))
                beta = torch.cat((beta_copy, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, p_mask, l, a, beta



    def cell_division(self, x, p, q, p_mask, l, a, beta_decay = 1.0):

        if self.recruitment_sim:
            num_compleated_cells = torch.sum(p_mask == 4)
            total_renal_cells = num_compleated_cells + torch.sum(p_mask == 3) 
            if num_compleated_cells >= self.recruitment_start and total_renal_cells < self.recruitment_stop:
                division, x, p, q, p_mask, l, a, beta = self.recruitment_sim_func(x, p, q, p_mask, l, a)
                return division, x, p, q, p_mask, l, a, beta
            else:
                return False, x, p, q, p_mask, l, a, self.beta_tensor
            
        beta = self.beta_tensor
        dt = self.dt

        if torch.sum(beta) < 1e-5:
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

def save(data_tuple, name, output_folder):
    with open(f'{output_folder}/{name}.npy', 'wb') as f:
        pickle.dump(data_tuple, f)

def run_simulation(sim_dict):
    # Make the simulation runner object:
    data_tuple = sim_dict.pop('data')
    verbose    = sim_dict.pop('verbose')
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every']

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

    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    p_mask_lst = [p_mask]

    with open(f'data/' + output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder='data/' + output_folder)


    notes = sim_dict['notes']
    if verbose:
        print('Starting simulation with notes:')
        print(notes)

    i = 0
    t1 = time()

    for xx, pp, qq, pp_mask in itertools.islice(runner, yield_steps):
        i += 1
        if verbose:
            print(f'Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end='\r')

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        p_mask_lst.append(pp_mask)
        
        if len(xx) > sim_dict['max_cells']:
            break

        if i % 100 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder='data/' + output_folder)

    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder='data/' + output_folder)