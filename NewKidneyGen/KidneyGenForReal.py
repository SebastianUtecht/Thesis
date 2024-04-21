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
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)
        self.UEC_l0    = torch.tensor(sim_dict['UEC_l0'], device=self.device, dtype=self.dtype)
        self.MPC_iREC_l0    = torch.tensor(sim_dict['MPC_iREC_l0'], device=self.device, dtype=self.dtype)
        self.pUEC_iREC_l0   = torch.tensor(sim_dict['pUEC_iREC_l0'], device=self.device, dtype=self.dtype)
        self.seethru        = sim_dict['seethru']
        self.seethru_stop   = sim_dict['seethru_stop']
        self.UEC_alpha, self.vesicle_alphas, self.tube_alphas = sim_dict['alphas'] 
        self.alphas         = torch.tensor([0,self.UEC_alpha,0,self.vesicle_alphas[0],self.vesicle_alphas[1]], device=self.device, dtype=self.dtype)
        self.betas          = torch.tensor(sim_dict['betas'], device=self.device, dtype=self.dtype)
        self.activation_loc = torch.tensor(sim_dict['activation_sphere'][0], device=self.device, dtype=self.dtype)
        self.activation_radius = sim_dict['activation_sphere'][1]
        self.pit_loc = torch.tensor(sim_dict['pit_sphere'][0], device=self.device, dtype=self.dtype)
        self.pit_radius = sim_dict['pit_sphere'][1]
        self.REC_IC_ratio   = sim_dict['REC_IC_ratio']
        self.max_cells      = sim_dict['max_cells']
        self.prolif_delay   = sim_dict['prolif_delay']
        self.abs_s2s3       = sim_dict['abs_s2s3']
        self.min0_s1        = sim_dict['min0_s1']
        self.yield_every    = sim_dict['yield_every']
        self.random_seed    = sim_dict['random_seed']
        self.vesicle_time   = sim_dict['vesicle_time']
        self.UEC_bound_strength = sim_dict['UEC_bound_strength']
        self.avg_q          = sim_dict['avg_q']
        self.iREC_evo, self.iREC_evo_tot_time, self.polar_time  = self.make_iREC_evolution(sim_dict['iREC_time_arr'])
        self.recruitment_stop = sim_dict['recruitment_stop'] 
        self.iREC_idx  = torch.tensor([0],dtype=torch.int, device=self.device)
        self.iREC_timing = torch.tensor([0],dtype=torch.int, device=self.device)

        self.vesicle_formation = False
        self.tube_formation    = False
        self.beta_tensor = None 
        self.d = None
        self.idx = None

        torch.manual_seed(self.random_seed)

    def torch_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool, device=self.device)
        mask[indices] = False
        return tensor[mask]
    
    def make_iREC_evolution(self, time_arr):
        i_pref_time     = int(time_arr[0] / self.dt)
        pref_time       = int(time_arr[1] / self.dt)
        i_polar_time    = int(time_arr[2] / self.dt)
        tot_time        = i_pref_time + pref_time + i_polar_time 

        non_polar_ipref = torch.linspace(self.lambdas[0][0], 1, i_pref_time)
        non_polar_pref  = torch.ones(pref_time)
        non_polar_ipolar= torch.linspace(1,0,i_polar_time)
        non_polar_part  = torch.cat((non_polar_ipref, non_polar_pref, non_polar_ipolar), axis=0)
        non_polar_part = non_polar_part.to(self.device)

        polar_end_vals  = self.lambdas[-1][1:]
        polar_ipref     = torch.zeros((i_pref_time + pref_time,3), device=self.device)
        polar_ipolar    = torch.linspace(0,1, i_polar_time)
        polar_ipolar    = polar_end_vals[None:,] * polar_ipolar[:,None].to(self.device)
        polar_part      = torch.cat((polar_ipref, polar_ipolar) , axis=0)

        total = torch.cat((non_polar_part[:,None],polar_part), axis=1)
        return total, tot_time, (i_pref_time + pref_time + 1)

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
    
    def init_polarity(self, x, p, q, p_mask, z_mask, idx):

        with torch.no_grad():
            iREC_neighbors = idx[self.iREC_idx][self.iREC_timing == self.polar_time]
            if len(iREC_neighbors) > 0:
                iREC_neighbors_mask = torch.logical_or(p_mask[iREC_neighbors] == 3, p_mask[iREC_neighbors] == 4)
                iREC_neighbors_mask *= z_mask[self.iREC_idx][self.iREC_timing == self.polar_time]
                iREC_neighbors_pos  = x[iREC_neighbors] * iREC_neighbors_mask[:,:,None].expand(iREC_neighbors.shape[0], idx.shape[1], 3)
                iREC_polar_centers  = ( (iREC_neighbors_pos.sum(axis=1) + x[self.iREC_idx][self.iREC_timing == self.polar_time]) /
                                        (iREC_neighbors_pos.sum(axis=2) != 0).sum(axis=1).reshape(-1,1) + 1 )
                
                init_p = x[self.iREC_idx][self.iREC_timing == self.polar_time] - iREC_polar_centers
                init_p /= torch.sqrt(torch.sum(init_p ** 2))
                init_q = torch.empty_like(init_p).normal_()
                init_q -= torch.sum(init_q * init_p) * init_p
                init_q /= torch.sqrt(torch.sum(init_q ** 2))

                p[self.iREC_idx][self.iREC_timing == self.polar_time] = init_p
                q[self.iREC_idx][self.iREC_timing == self.polar_time] = init_q
    
    def UEC_bound_potential(self, x, p_mask):
        UEC_cells = x[torch.logical_or(p_mask == 1, p_mask == 5)]
        trans_UEC_cells = (self.UEC_inv_lin_trans @ UEC_cells[:,:,None]).squeeze()
        dists = (self.UEC_x0_transformed[:,0] - trans_UEC_cells[:,0])
        dists = torch.max(dists, torch.zeros_like(dists, device=self.device))
        pot = torch.sum(self.UEC_bound_strength * torch.exp(dists**2))
        return pot


    def potential(self, x, p, q, p_mask, l, a, idx, d):
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

        self.init_polarity(x, p, q, p_mask, z_mask, idx)

        interaction_mask = torch.cat((p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])[:,:,None], p_mask[idx][:,:,None]), dim=2)

        MPC_iREC_mask = torch.sum(interaction_mask == torch.tensor([0,3], device=self.device), dim=2) == 2
        MPC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,0], device=self.device), dim=2) == 2, MPC_iREC_mask)

        pUEC_iREC_mask = torch.sum(interaction_mask == torch.tensor([5,3], device=self.device), dim=2) == 2
        pUEC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,5], device=self.device), dim=2) == 2, MPC_iREC_mask)

        pUEC_UEC_mask = torch.sum(interaction_mask == torch.tensor([5,1], device=self.device), dim=2) == 2
        pUEC_UEC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([1,5], device=self.device), dim=2) == 2, MPC_iREC_mask)

        UEC_iREC_mask = torch.sum(interaction_mask == torch.tensor([1,3], device=self.device), dim=2) == 2
        UEC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([3,1], device=self.device), dim=2) == 2, MPC_iREC_mask)

        pUEC_iREC_mask = torch.sum(interaction_mask == torch.tensor([5,1], device=self.device), dim=2) == 2
        pUEC_iREC_mask = torch.logical_or( torch.sum(interaction_mask == torch.tensor([1,5], device=self.device), dim=2) == 2, MPC_iREC_mask)

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        l_i     = l[:, None, :].expand(p.shape[0], idx.shape[1], -1)
        l_j     = l[idx, :]
        l_min   = torch.min(l_i, l_j)
        l_min[MPC_iREC_mask] = torch.tensor([self.MPC_iREC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)
        l_min[pUEC_UEC_mask] *= 0.9
        l_min[torch.sum(l_min, dim=2) == 0] = torch.tensor([self.UEC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)
        l_min[UEC_iREC_mask] *= .5
        l_min[pUEC_iREC_mask] *= 0.7                #[:,0] = torch.tensor([self.pUEC_iREC_l0, 0, 0, 0], dtype=self.dtype, device=self.device)

        a_i = a[:, None].expand(p.shape[0], idx.shape[1])
        a_j = a[idx]
        a_min = torch.min(a_i, a_j)[:,:,None].expand(a_i.shape[0], a_i.shape[1], 3)

        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
        qj = q[idx]

        if self.vesicle_formation:
            pi_tilde = pi - a_min*dx
            pj_tilde = pj + a_min*dx

            pi_tilde = pi_tilde/torch.sqrt(torch.sum(pi_tilde ** 2, dim=2))[:, :, None]               # The p-tildes are normalized
            pj_tilde = pj_tilde/torch.sqrt(torch.sum(pj_tilde ** 2, dim=2))[:, :, None]

        elif self.tube_formation:
            
            if self.avg_q:
                avg_q = (qi + qj)*0.5
            else:
                avg_q = qi

            ts = (avg_q * dx).sum(axis = 2)

            angle_dx = avg_q * ts[:,:,None]

            if self.seethru != 0:
                angle_dx = angle_dx * d[:,:,None]                                                        #Use if using seethru != 0
                    
            pi_tilde = pi + a_min*angle_dx
            pj_tilde = pj - a_min*angle_dx
        else:
            pi_tilde = pi
            pj_tilde = pj

        S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
        S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)            # Calculating S2 (The ABP-PCP part of S).
        S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

        if self.abs_s2s3:
            S2 = torch.abs(S2)
            S3 = torch.abs(S3)
        if self.min0_s1:
            S1[S1 < 0] = 0.0 
        if self.vesicle_formation:
            l_min[:,:,3] = 0.0

        S = l_min[:,:,0] + l_min[:,:,1] * S1 + l_min[:,:,2] * S2 + l_min[:,:,3] * S3
        
        Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
        Vij_sum = torch.sum(Vij)
        if self.UEC_bound_strength:
            pot = self.UEC_bound_potential(x, p_mask)
            # print(pot)
            Vij_sum += pot

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

        for i in range(len(self.lambdas)):
            l[p_mask == i] = self.lambdas[i]
            a[p_mask == i] = self.alphas[i]

        pUEC = torch.sum(( self.pit_loc -  x)**2, dim=1) < self.pit_radius**2
        p_mask[torch.logical_and(pUEC, p_mask == 1)] = 5


        if self.UEC_bound_strength:
            with torch.no_grad():
                # We find all the polar particles
                p_polar = p[torch.logical_or(p_mask == 1, p_mask == 5)].clone().squeeze()
                q_polar = q[torch.logical_or(p_mask == 1, p_mask == 5)].clone().squeeze()

                # Gram-Schmidt orthanormalization
                p_polar /= torch.sqrt(torch.sum(p_polar ** 2, dim=1))[:, None]
                q_polar -= torch.sum(q_polar * p_polar, dim=1)[:,None] * p_polar
                q_polar /= torch.sqrt(torch.sum(q_polar ** 2, dim=1))[:, None]

                # Matrix for linear transformation
                lin_trans = torch.zeros((len(p_polar), 3, 3), device=self.device, dtype=self.dtype)
                lin_trans[:,:,0] = p_polar
                lin_trans[:,:,1] = q_polar
                lin_trans[:,:,2] = torch.cross(p_polar,q_polar, dim=1)
                
                print(lin_trans.shape)

                self.UEC_inv_lin_trans  = torch.linalg.inv(lin_trans)
                print(self.UEC_inv_lin_trans.shape)
                print(x[torch.logical_or(p_mask == 1, p_mask == 5)].shape)
                self.UEC_x0_transformed = (self.UEC_inv_lin_trans @ x[torch.logical_or(p_mask == 1, p_mask == 5)][:,:,None]).squeeze()

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
        elif self.idx is None:
            return True
        if self.tube_formation:
            alt_tstep = tstep - self.vesicle_time
        else:
            alt_tstep = tstep

        n_update = 1 if alt_tstep < 100 else max([1, int(20 * np.tanh(alt_tstep / 200))])
        return (tstep % n_update == 0)

    def time_step(self, x, p, q, p_mask, l, a, tstep):

        if tstep < self.vesicle_time:
            self.vesicle_formation = True
        else:
            self.vesicle_formation = False
            self.tube_formation    = True
            self.alphas[-3:-1] = torch.tensor(self.tube_alphas, device=self.device, dtype=self.dtype) 

        if tstep == (self.prolif_delay + 1):
            if torch.sum(self.betas) > 0:
                for i in range(len(self.betas)):
                    self.beta_tensor[p_mask == i] = self.betas[i]

        if tstep == (self. seethru_stop + 1):
            self.seethru = 0

        # Start with cell division
        division, x, p, q, p_mask, self.beta = self.cell_division(x, p, q, p_mask)

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
            cells_w_polarity = torch.sum(l[:,1:], dim=1) > 0.0
            p[cells_w_polarity] /= torch.sqrt(torch.sum(p[cells_w_polarity] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
            q[cells_w_polarity] /= torch.sqrt(torch.sum(q[cells_w_polarity] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

        # Calculate potential
        V, self.true_neighbour_max = self.potential(x, p, q, p_mask, l, a, idx, d)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x += -x.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            p += -p.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            q += -q.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            p.grad.zero_()
            q.grad.zero_()

        x.grad.zero_()

        if len(self.iREC_timing) > 1:
            self.iREC_timing += 1
            fin_REC_bool = self.iREC_timing >= self.iREC_evo_tot_time
            if torch.any(fin_REC_bool):
                fin_REC_idx = self.iREC_idx[fin_REC_bool]
                p_mask[fin_REC_idx] = 4
                removal_idx = torch.argwhere(torch.isin(self.iREC_idx, fin_REC_idx))[:,0]
                self.iREC_idx = self.torch_delete(self.iREC_idx, removal_idx)
                self.iREC_timing = self.torch_delete(self.iREC_timing, removal_idx)
                l[fin_REC_idx] = self.lambdas[4]
                a[fin_REC_idx] = self.alphas[4]
            l[self.iREC_idx] = self.iREC_evo[self.iREC_timing]

        #Check for new types
        if tstep < self.recruitment_stop:
            cells_within_act = torch.sum(( self.activation_loc -  x)**2, dim=1) < self.activation_radius**2
            changing_cells   = torch.logical_and(cells_within_act, (p_mask == 0))
            if not(torch.any(changing_cells)):
                return x, p, q, p_mask, l, a
            else: 
                changing_cells_idx = torch.argwhere(changing_cells)[:,0]
                REC_IC_mask = torch.rand(len(changing_cells_idx), device=self.device) < self.REC_IC_ratio
                iREC_cells = changing_cells_idx[REC_IC_mask]
                IC_cells = changing_cells_idx[~REC_IC_mask]
                p_mask[iREC_cells] = 3
                p_mask[IC_cells] = 2

                if not(torch.any(self.iREC_idx)):
                    self.iREC_idx = iREC_cells.clone()
                    self.iREC_timing = torch.zeros_like(iREC_cells)
                else:
                    self.iREC_idx = torch.cat((self.iREC_idx,iREC_cells), dim=0)
                    self.iREC_timing = torch.cat((self.iREC_timing,torch.zeros_like(iREC_cells)), dim=0)
                
                l[IC_cells] = self.lambdas[2] 
                a[IC_cells] = self.alphas[2]
                a[self.iREC_idx] = self.alphas[3]

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

    def cell_division(self, x, p, q, p_mask, beta_decay = 1.0):

        beta = self.beta_tensor
        dt = self.dt

        if torch.sum(beta) < 1e-5:
            return False, x, p, q, p_mask, beta

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
                p_mask0 = p_mask[idx]
                b0      = beta[idx] * beta_decay

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
                p_mask = torch.cat((p_mask, p_mask0))
                beta = torch.cat((beta, b0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, p_mask, beta

def save(data_tuple, name, output_folder):
    with open(f'{output_folder}/{name}.npy', 'wb') as f:
        pickle.dump(data_tuple, f)

def run_simulation(sim_dict):
    # Make the simulation runner object:
    data_tuple = sim_dict.pop('data')
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
    print('Starting simulation with notes:')
    print(notes)

    i = 0
    t1 = time()

    for xx, pp, qq, pp_mask in itertools.islice(runner, yield_steps):
        i += 1
        print(f'Running {i} of {yield_steps}   ({yield_every * i} of {yield_every * yield_steps})   ({len(xx)} cells)', end='\r')

        x_lst.append(xx)
        p_lst.append(pp)
        q_lst.append(qq)
        p_mask_lst.append(pp_mask)
        
        if len(xx) > sim_dict['max_cells']:
            break

        if i % 100 == 0:
            save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder='data/' + output_folder)

    print(f'Simulation done, saved {yield_steps} datapoints')
    print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder='data/' + output_folder)