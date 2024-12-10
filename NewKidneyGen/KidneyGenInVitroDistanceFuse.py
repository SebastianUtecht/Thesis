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
import networkx as nx
from DataGeneration import *

class Simulation:
    def __init__(self, sim_dict):
        self.device         = sim_dict['device']
        self.dtype          = sim_dict['dtype']
        self.k              = sim_dict['init_k'] 
        self.true_neighbour_max     = sim_dict['init_k']//2
        self.dt             = sim_dict['dt']
        self.sqrt_dt        = np.sqrt(self.dt)
        self.stroma_eta     = sim_dict['stroma_eta']
        self.NPC_eta        = sim_dict['NPC_eta']    
        self.lambdas        = torch.tensor(sim_dict['lambdas'], device=self.device, dtype=self.dtype)
        self.pre_lambdas    = sim_dict['pre_lambdas']
        self.gamma          = sim_dict['gamma']
        self.warmup_dur     = sim_dict['warmup_dur']
        self.pre_polar_dur  = sim_dict['pre_polar_dur']
        self.seethru        = sim_dict['seethru']
        self.vesicle_alpha  = sim_dict['vesicle_alpha']
        self.tube_alpha     = sim_dict['tube_alpha']
        self.crit_ves_size  = sim_dict['crit_ves_size']
        self.min_ves_time   = sim_dict['min_ves_time']
        self.max_cells      = sim_dict['max_cells']
        self.prolif_delay   = sim_dict['prolif_delay']
        self.abs_s2s3       = sim_dict['abs_s2s3']
        self.tube_wall_str  = sim_dict['tube_wall_str']
        self.new_tube_wall= sim_dict['new_tube_wall']
        self.yield_every    = sim_dict['yield_every']
        self.bound_radius   = sim_dict['bound_radius']
        self.random_seed    = sim_dict['random_seed']
        self.avg_q          = sim_dict['avg_q']
        self.polar_initialization = sim_dict['polar_initialization']
        self.schausers_wall_mask = sim_dict['schausers_wall_mask']
        self.mean_cell_pr_dt = sim_dict['mean_cell_pr_dt']              #standard value of 0.15
        self.wall_mask_clusters = sim_dict['wall_mask_clusters']
        if self.mean_cell_pr_dt != 0.15:
            print('Note: mean_cell_pr_dt is not set to standard value of 0.15')

        self.all_tube       = False
        self.warming_up = False
        self.pre_polar  = False
        self.vesicle_formation = False
        self.tube_formation    = False
        self.proliffing_cells  = False
        self.d = None
        self.idx = None

        torch.manual_seed(self.random_seed)


        #For debugging

        self.seethru_counter = 1 
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
    
    @staticmethod
    def gauss_grad(d, dx, interaction_mask):
        with torch.no_grad():
            gauss   = torch.exp(-(d ** 2) * 0.04)
            gauss *= (interaction_mask == 2).type(torch.int)
            grad    = torch.sum((gauss[:, :, None] * dx * d[:,:,None]), dim=1)
        return grad
    
    def sphere_bound(self, pos):
        scaled_pos = pos / self.bound_radius
        scaled_dists   = torch.sum(scaled_pos**2, dim=1)
        v_add   = torch.where(scaled_dists > 1, (torch.exp(scaled_dists**2 - 1) - 1)*10, 0.)

        # make sure the number is not too large
        v_add = torch.where(v_add > 50., 50., v_add)

        v_add = torch.sum(v_add)

        return v_add

    def potential(self, x, p, q, p_mask, idx, d, tstep):

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

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        # Making interaction mask
        # interaction_mask = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1]) + p_mask[idx]
        interaction_mask = torch.cat((p_mask[:,None].expand(p_mask.shape[0], idx.shape[1])[:,:,None], p_mask[idx][:,:,None]), dim=2)
        
        # Mesenchyme - Mesenchyme mask
        MPC_MPC_mask = torch.sum(interaction_mask == torch.tensor([0,0], device=self.device), dim=2) == 2

        # Mesenchyme - intermediate (and not) renal epithelial mask
        MPC_iREC_mask   = torch.sum(interaction_mask == torch.tensor([0,1], device=self.device), dim=2) == 2
        MPC_iREC_mask   = torch.logical_or( torch.sum(interaction_mask == torch.tensor([1,0], device=self.device), dim=2) == 2, MPC_iREC_mask)
        MPC_REC_mask    = torch.sum(interaction_mask == torch.tensor([0,2], device=self.device), dim=2) == 2
        MPC_REC_mask    = torch.logical_or( torch.sum(interaction_mask == torch.tensor([2,0], device=self.device), dim=2) == 2, MPC_REC_mask)
        MPC_RECiREC_mask= torch.logical_or(MPC_iREC_mask, MPC_REC_mask)

        # Intermediate renal epithelial - intermediate (and not) renal epithelial mask
        iREC_iREC_mask  = torch.sum(interaction_mask == torch.tensor([1,1], device=self.device), dim=2) == 2
        iREC_REC_mask   = torch.sum(interaction_mask == torch.tensor([1,2], device=self.device), dim=2) == 2
        iREC_REC_mask   = torch.logical_or( torch.sum(interaction_mask == torch.tensor([2,1], device=self.device), dim=2) == 2, iREC_REC_mask)
        iREC_RECiREC_mask   = torch.logical_or(iREC_iREC_mask, iREC_REC_mask)

        # Renal epithelia - Renal epithelia mask
        REC_REC_mask    = torch.sum(interaction_mask == torch.tensor([2,2], device=self.device), dim=2) == 2
        polar_mask      = ~torch.any(interaction_mask == 0, dim=2)
    
        
        wall_mask = None

        # At vesicle initialization we make the APB point out
        if tstep == (self.warmup_dur + self.pre_polar_dur) and self.polar_initialization:
            temp_p_mask = p_mask.clone()
            temp_p_mask[temp_p_mask == 2] = 1
            with torch.no_grad():
                polar_neighbors = (temp_p_mask[idx][:,:,None].expand(temp_p_mask.shape[0], idx.shape[1],3)
                                * z_mask[:,:,None].expand(z_mask.shape[0], idx.shape[1],3) * x[idx])
                polar_centers = ((polar_neighbors.sum(axis=1) + x) / 
                                ((polar_neighbors.sum(axis=2) != 0).sum(axis=1).reshape(-1,1) + 1))
                p[:,:] = x - polar_centers
                p[temp_p_mask != 0] /= torch.sqrt(torch.sum(p[temp_p_mask != 0] ** 2, dim=1))[:, None]
                p[:,:] = torch.nan_to_num(p)

        # Calculate S
        if self.warming_up or self.pre_polar:
            if self.warming_up:
                MPC_MPC_l = MPC_RECiREC_l = iREC_RECiREC_l = REC_REC_l = self.lambdas[0][0].item()  #0.3 usually   
            else:
                MPC_MPC_l, MPC_RECiREC_l, iREC_RECiREC_l, REC_REC_l = self.pre_lambdas

            lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1]),
                    device=self.device)
            
            lam[MPC_MPC_mask] = torch.tensor(MPC_MPC_l, device=self.device, dtype=self.dtype) # Setting lambdas for pre non-polar interaction
            lam[MPC_RECiREC_mask] = torch.tensor(MPC_RECiREC_l, device=self.device, dtype=self.dtype) # Setting lambdas for pre polar-nonpolar interaction
            lam[iREC_RECiREC_mask] = torch.tensor(iREC_RECiREC_l, device=self.device, dtype=self.dtype) # Setting lambdas for pre pure polar interaction                                                              # We need these gradients in order to do backprob later.
            lam[REC_REC_mask] = torch.tensor(REC_REC_l, device=self.device, dtype=self.dtype) # Setting lambdas for pre pure polar interaction                                                              # We need these gradients in order to do backprob later.

            S = lam

        else:
            # Setting the lambdas
            MPC_MPC_ls       = self.lambdas[0]
            MPC_RECiREC_ls   = self.lambdas[1]
            iREC_RECiREC_ls  = self.lambdas[2]
            REC_REC_ls       = self.lambdas[3]

            lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                            device=self.device, dtype=self.dtype)
            
            lam[MPC_MPC_mask]       = MPC_MPC_ls
            lam[MPC_RECiREC_mask]   = MPC_RECiREC_ls
            lam[iREC_RECiREC_mask]  = iREC_RECiREC_ls
            lam[REC_REC_mask]       = REC_REC_ls

            # Expanding ABP and PCP for easy cross products
            pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
            pj = p[idx]
            qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
            qj = q[idx]

            # Setting the alphas
            alphas = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1]), device=self.device, dtype=self.dtype)
            alphas[iREC_RECiREC_mask] = self.vesicle_alpha
            alphas[REC_REC_mask]      = self.tube_alpha
            alphas = alphas[:,:,None].expand(alphas.shape[0], alphas.shape[1], 3)

            # Using avg_q?
            if self.avg_q:
                avg_q = (qi + qj)*0.5
            else:
                avg_q = qi

            # Calculating the anisotropic angle_dx
            ts = (avg_q * dx).sum(axis = 2)
            angle_dx = avg_q * ts[:,:,None]

            # Calculating the isotropic angle
            angle_dx[~REC_REC_mask] = dx[~REC_REC_mask]

            # Addition if we use seethru
            if self.seethru != 0:
                angle_dx[REC_REC_mask] = angle_dx[REC_REC_mask] * d[:,:,None][REC_REC_mask]

            # Permute the ABPs such that we get wedging
            pi_tilde = pi - alphas * angle_dx
            pj_tilde = pj + alphas * angle_dx

            # The permuted ABPs are normalized
            interactions_w_polarity = torch.logical_or(iREC_RECiREC_mask, REC_REC_mask)
            pi_tilde[interactions_w_polarity] = pi_tilde[interactions_w_polarity]/torch.sqrt(torch.sum(pi_tilde[interactions_w_polarity] ** 2, dim=1))[:, None] 
            pj_tilde[interactions_w_polarity] = pj_tilde[interactions_w_polarity]/torch.sqrt(torch.sum(pj_tilde[interactions_w_polarity] ** 2, dim=1))[:, None]

            # All the S-terms are calculated
            S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
            S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)            # Calculating S2 (The ABP-PCP part of S).
            S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

            # Nematic PCP?
            if self.abs_s2s3:
                S2 = torch.abs(S2)
                S3 = torch.abs(S3)

            # Inducing non-polar interaction between cell walls

            if self.tube_wall_str != None:
                with torch.no_grad():

                    if self.schausers_wall_mask:
                        wall_mask = (torch.sum(pi * pj , dim = 2) < 0.0) * (torch.sum(-dx * pj , dim = 2) < 0.0)
                    else:
                        wall_mask = torch.sum(pi * pj , dim = 2) < 0.0

                    wall_mask = torch.logical_and(wall_mask , polar_mask)
                    lam[wall_mask] = torch.tensor([self.tube_wall_str, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
            # Calculating S
            S = lam[:,:,0] + lam[:,:,1] * S1 + lam[:,:,2] * S2 + lam[:,:,3] * S3

        # # Tube-wall which only facilitates tubes pushing off each other
        # if (self.new_tube_wall and self.tube_wall_str == None) and (not(self.warming_up) and not(self.pre_polar)):
        #     wall_mask = torch.sum(pi * pj , dim = 2) < 0.0
        #     wall_mask = torch.logical_and(wall_mask , REC_REC_mask)
        #     Vij = z_mask[~wall_mask].float() * (torch.exp(-d[~wall_mask]) - S[~wall_mask] * torch.exp(-d[~wall_mask]/5))

        #     wall_potential = torch.where(d[wall_mask] < 2.5 , 1 * (2.5 - d[wall_mask])**2, 0.0)  #2.011 is the relaxation distance

        #     Vij_sum = torch.sum(Vij) + torch.sum(wall_potential)

        #else:

        Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
        Vij_sum = torch.sum(Vij)

        # Utilize spherical boundary conditions?
        if self.bound_radius:
            bc = self.sphere_bound(x)
        else:
            bc = 0.
        # Direct ABPs away from center of mass?
        if (not self.warming_up and not self.pre_polar) and self.gamma:
            gauss_grad = self.gauss_grad(d, dx, interaction_mask)
            Vi     = torch.sum(self.gamma * p * gauss_grad, dim=1)
            return Vij_sum - torch.sum(Vi) + bc , int(m), z_mask, idx, wall_mask
        else:
            return Vij_sum + bc, int(m), z_mask, idx, wall_mask

    def init_simulation(self, x, p, q, p_mask):
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)

        return x, p, q, p_mask
    
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
        elif tstep < self.warmup_dur:
            alt_tstep = tstep
        elif tstep < (self.pre_polar_dur + self.warmup_dur):
            alt_tstep = tstep - self.warmup_dur
        else:
            alt_tstep = tstep - (self.warmup_dur + self.pre_polar_dur)

        n_update = 1 if alt_tstep < 50 else max([1, int(20 * np.tanh(alt_tstep / 200))])

        return (tstep % n_update == 0)

    def time_step(self, x, p, q, p_mask, tstep):
        if tstep < self.warmup_dur:
            self.warming_up = True
        elif tstep < (self.pre_polar_dur + self.warmup_dur):
            self.warming_up = False
            self.pre_polar  = True
        elif tstep > (self.pre_polar_dur + self.warmup_dur) and not(self.all_tube):
            self.vesicle_formation = True
            self.seethru = 0  
            self.warming_up = False
            self.pre_polar = False
            if torch.sum(p_mask == 1) == 0:
                self.all_tube = True
        else:
            self.pre_polar          = False
            self.vesicle_formation  = False
            self.warming_up         = False
            self.tube_formation     = True

        if tstep == (self.warmup_dur + self.pre_polar_dur + self.prolif_delay + 1):
            if self.mean_cell_pr_dt > 0:
                self.proliffing_cells = True

        # Start with cell division
        division, x, p, q, p_mask = self.cell_division(x, p, q, p_mask)
        
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
        
        if not(self.warming_up) and not(self.pre_polar):

            with torch.no_grad():
                p[p_mask != 0] /= torch.sqrt(torch.sum(p[p_mask != 0] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
                q[p_mask != 0] /= torch.sqrt(torch.sum(q[p_mask != 0] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

                p[p_mask == 0] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)
                q[p_mask == 0] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)

        # Calculate potential
        V, self.true_neighbour_max, z_mask, idx, wall_mask = self.potential(x, p, q, p_mask, idx, d, tstep)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            if not(self.warming_up) and not(self.pre_polar):
                stroma_mask = (p_mask == 0)
                NPC_mask    = ~stroma_mask
                x[NPC_mask] += -x.grad[NPC_mask] * self.dt + self.NPC_eta * torch.empty(*x[NPC_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                x[stroma_mask] += -x.grad[stroma_mask] * self.dt + self.stroma_eta * torch.empty(*x[stroma_mask].shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

                p += -p.grad * self.dt + self.NPC_eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                q += -q.grad * self.dt + self.NPC_eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

                p.grad.zero_()
                q.grad.zero_()
            else:
                x += -x.grad * self.dt + self.stroma_eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

        x.grad.zero_()

        # Only do this every N timesteps
        # Only do if vesicle_formation == True
        min_ves_time_bool = tstep > (self.pre_polar_dur + self.warmup_dur + self.min_ves_time)
        if self.update_neighbors_bool(tstep, division) and (self.vesicle_formation and min_ves_time_bool):

            # Append the name of the nodes themselves
            if self.wall_mask_clusters:
                used_z_mask = z_mask * (~wall_mask)
            else:
                used_z_mask = z_mask

            extended_idx    = torch.cat( (torch.arange(len(idx), device=self.device)[:,None] , idx ), dim=1)
            extended_z_mask = torch.cat( (torch.ones(len(idx), device=self.device)[:,None] , used_z_mask ), dim=1)

            # Transition to cpu
            polar_idx    = extended_idx[p_mask != 0]
            polar_z_mask = extended_z_mask[p_mask != 0]
            polar_p_mask = torch.where(p_mask == 2, 1, p_mask)
            
            # Find graph structure of polar particles
            edge_lst= get_edge_lst(polar_idx, p_mask=polar_p_mask, z_mask=polar_z_mask  )
            graph   = make_graph(edge_lst.detach().cpu().numpy())

            # Find sizes of connected components
            # If connected component size > threshold --> p_mask = 2 --> Tube
            for comp in nx.connected_components(graph):
                if len(comp) > self.crit_ves_size:
                    p_mask[list(comp)] = 2

        return x, p, q, p_mask

    def simulation(self, x, p, q, p_mask):
        
        x, p, q, p_mask = self.init_simulation(x, p, q, p_mask)

        tstep = 0
        while True:
            tstep += 1
            x, p, q, p_mask = self.time_step(x, p, q, p_mask, tstep)

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

        if not(self.proliffing_cells):
            return False, x, p, q, p_mask

        division = torch.rand(1).item() < self.mean_cell_pr_dt

        if division:
            with torch.no_grad():
                NPC_idxs = torch.nonzero(~(p_mask == 0))[:, 0]
                idx = NPC_idxs[torch.randint(0, len(NPC_idxs), (1,)).item()]

                x0      = x[idx, :]
                p0      = p[idx, :]
                q0      = q[idx, :]
                p_mask0 = p_mask[idx]

                # make a random vector and normalize to get a random direction
                move = torch.empty_like(x0).normal_()
                move[p_mask0 == 1] = self.get_prolif_positions(p0, q0, p_mask0, mask_ind=1)
                move[p_mask0 == 2] = self.get_prolif_positions(p0, q0, p_mask0, mask_ind=2)
                
                if torch.numel(move) == 3:
                    move /= torch.sqrt(torch.sum(move**2))
                else:
                    move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

                # place new cells
                x0 = x0 + move

                # append new cell data to the system state
                if torch.numel(x0) == 3:
                    x = torch.cat((x, x0[None, :]))
                    p = torch.cat((p, p0[None, :]))
                    q = torch.cat((q, q0[None, :]))
                    p_mask = torch.cat((p_mask, p_mask0[None]))
                else:
                    x = torch.cat((x, x0))
                    p = torch.cat((p, p0))
                    q = torch.cat((q, q0))
                    p_mask = torch.cat((p_mask, p_mask0))

        x.requires_grad = True
        p.requires_grad = True
        q.requires_grad = True

        return division, x, p, q, p_mask

def get_edge_lst(adj_arr, z_mask, p_mask):
    adj_arr[z_mask == 0] = -1
    polar_idx = torch.argwhere(p_mask == 1).squeeze()
    non_polar_mask = torch.any(adj_arr[:,:,None] == polar_idx, dim=2)
    double_mask = non_polar_mask * z_mask

    masked_adj_arr = adj_arr * double_mask
    edge_lst = torch.cat((adj_arr[:,0][:,None].expand((adj_arr.shape[0], adj_arr.shape[1])).reshape(-1,1) , masked_adj_arr.reshape(-1,1)), dim=1) 
    edge_lst = edge_lst[edge_lst[:,1] != 0].type(torch.int)

    return edge_lst

def make_graph(edge_lst):
    G = nx.from_edgelist(edge_lst)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def save(data_tuple, name, output_folder):
    with open(f'{output_folder}/{name}.npy', 'wb') as f:
        pickle.dump(data_tuple, f)

def run_simulation(sim_dict):
    # Make the simulation runner object:
    verbose    = sim_dict.pop('verbose')
    x_dist = sim_dict['x_dist']
    N_stroma = sim_dict['N_stroma']
    yield_steps, yield_every = sim_dict['yield_steps'], sim_dict['yield_every']

    data = np.load('single_vesicle_data.npy', allow_pickle=True)
    p_mask0, x0, p0, q0 = data

    #copy for second vesicle
    x1 = np.copy(x0)
    p1= np.copy(p0)
    q1 = np.copy(q0)
    p_mask1 = np.copy(p_mask0)
    print(len(x0))

    #translating the vesicles and stitching the datasets together
    x0[:,0] -= x_dist/2
    x1[:,0] += x_dist/2

    #remove half of x0
    x0 = x0[:len(x0)//2]
    p0 = p0[:len(p0)//2]
    q0 = q0[:len(q0)//2]
    p_mask0 = p_mask0[:len(p_mask0)//2]

    x = np.concatenate((x0, x1))
    p = np.concatenate((p0, p1))
    q = np.concatenate((q0, q1))
    p_mask = np.concatenate((p_mask0, p_mask1))

    if N_stroma > 0:
        # Make sure the stroma is not too close to the vesicles
        # find all stroma that is closer to a vesicle cell than 3
        p_mask_stroma, x_stroma, p_stroma, q_stroma = make_random_sphere(N_stroma, 1, radius=x_dist*1.5)
        p_stroma = p_stroma[~np.any(np.linalg.norm(x_stroma[:, None] - x, axis=2) < 3, axis=1)]
        q_stroma = q_stroma[~np.any(np.linalg.norm(x_stroma[:, None] - x, axis=2) < 3, axis=1)]
        p_mask_stroma = p_mask_stroma[~np.any(np.linalg.norm(x_stroma[:, None] - x, axis=2) < 3, axis=1)]
        x_stroma = x_stroma[~np.any(np.linalg.norm(x_stroma[:, None] - x, axis=2) < 3, axis=1)]
        print('Stroma cells:', len(x_stroma))
        
        x = np.concatenate((x, x_stroma))
        p = np.concatenate((p, p_stroma))
        q = np.concatenate((q, q_stroma))
        p_mask = np.concatenate((p_mask, p_mask_stroma))

    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)

    output_folder = sim_dict['output_folder']

    try: 
        os.mkdir(output_folder)
    except:
        pass

    x_lst = [x]
    p_lst = [p]
    q_lst = [q]
    p_mask_lst = [p_mask]

    with open(output_folder + '/sim_dict.json', 'w') as f:
        sim_dict['dtype'] = str(sim_dict['dtype'])
        json.dump(sim_dict, f, indent = 2)

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)


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
            save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)
    
    if verbose:
        print(f'Simulation done, saved {i} datapoints')
        print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)