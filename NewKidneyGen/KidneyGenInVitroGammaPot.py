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

class Simulation:
    def __init__(self, sim_dict):
        self.device         = sim_dict['device']
        self.dtype          = sim_dict['dtype']
        self.k              = sim_dict['init_k'] 
        self.true_neighbour_max     = sim_dict['init_k']//2
        self.dt             = sim_dict['dt']
        self.sqrt_dt        = np.sqrt(self.dt)
        self.eta            = sim_dict['eta']
        self.lambdas        = sim_dict['lambdas']
        self.pre_lambdas    = sim_dict['pre_lambdas']
        self.gamma          = sim_dict['gamma']
        self.warmup_dur     = sim_dict['warmup_dur']
        self.pre_polar_dur  = sim_dict['pre_polar_dur']
        self.seethru        = sim_dict['seethru']
        self.vesicle_alpha  = sim_dict['vesicle_alpha']
        self.tube_alpha     = sim_dict['tube_alpha']
        self.alpha = self.vesicle_alpha
        self.non_polar_prolif_rate  = sim_dict['betas'][0]
        self.polar_prolif_rate      = sim_dict['betas'][1]
        self.max_cells      = sim_dict['max_cells']
        self.prolif_delay   = sim_dict['prolif_delay']
        self.abs_s2s3       = sim_dict['abs_s2s3']
        self.cell_wall_str  = sim_dict['cell_wall_str']
        self.newnew_cellwall= sim_dict['newnew_cellwall']
        self.yield_every    = sim_dict['yield_every']
        self.bound_radius   = sim_dict['bound_radius']
        self.random_seed    = sim_dict['random_seed']
        self.vesicle_time   = sim_dict['vesicle_time']
        self.avg_q          = sim_dict['avg_q']
        self.polar_initialization = sim_dict['polar_initialization']

        self.warming_up = False
        self.pre_polar  = False
        self.vesicle_formation = False
        self.tube_formation    = False
        self.beta = None 
        self.d = None
        self.idx = None

        torch.manual_seed(self.random_seed)

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
        interaction_mask = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1]) + p_mask[idx]

        if tstep == (self.warmup_dur + self.pre_polar_dur) and self.polar_initialization:
            with torch.no_grad():
                polar_neighbors = (p_mask[idx][:,:,None].expand(p_mask.shape[0], idx.shape[1],3)
                                * z_mask[:,:,None].expand(z_mask.shape[0], idx.shape[1],3) * x[idx])
                polar_centers = ((polar_neighbors.sum(axis=1) + x) / 
                                ((polar_neighbors.sum(axis=2) != 0).sum(axis=1).reshape(-1,1) + 1))
                p[:,:] = x - polar_centers
                p[p_mask != 0] /= torch.sqrt(torch.sum(p[p_mask != 0] ** 2, dim=1))[:, None]
                p[:,:] = torch.nan_to_num(p)

        # Calculate S
        if self.warming_up or self.pre_polar:
            if self.warming_up:
                pre_l_00 = pre_l_01 = pre_l_11 = 1.
            else:
                pre_l_00, pre_l_01, pre_l_11 = self.pre_lambdas

            lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1]),
                    device=self.device)                                                               # Initializing an empty array for our pre-lambdas
            lam[interaction_mask == 0] = torch.tensor(pre_l_00, device=self.device, dtype=self.dtype) # Setting lambdas for pre non-polar interaction
            lam[interaction_mask == 1] = torch.tensor(pre_l_01, device=self.device, dtype=self.dtype) # Setting lambdas for pre polar-nonpolar interaction
            lam[interaction_mask == 2] = torch.tensor(pre_l_11, device=self.device, dtype=self.dtype) # Setting lambdas for pre pure polar interaction                                                              # We need these gradients in order to do backprob later.

            S = lam
        else:
            l00, l01, l1, l2, l3 = self.lambdas

            pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
            pj = p[idx]
            qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
            qj = q[idx]

            lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                            device=self.device, dtype=self.dtype)                                                            # Initializing an empty array for our lambdas
            lam[interaction_mask == 0] = torch.tensor([l00,0,0,0], device=self.device,  dtype=self.dtype)                     # Setting lambdas for non polar interaction
            lam[interaction_mask == 1] = torch.tensor([l01,0,0,0], device=self.device,  dtype=self.dtype)                     # Setting lambdas for polar-nonpolar interaction
            lam[interaction_mask == 2] = torch.tensor([0,l1,l2,l3], device=self.device, dtype=self.dtype)                    # Setting lambdas for pure polar interaction

            if self.alpha != 0 and self.vesicle_formation:
                pi_tilde = pi - self.alpha*dx
                pj_tilde = pj + self.alpha*dx

            elif self.alpha !=0 and self.tube_formation:
                alpha = self.alpha
                
                if self.avg_q:
                    avg_q = (qi + qj)*0.5
                else:
                    avg_q = qi

                ts = (avg_q * dx).sum(axis = 2)

                angle_dx = avg_q * ts[:,:,None]
                # angle_dx = angle_dx * d[:,:,None]                                                        #Use if using seethru != 0
                        
                pi_tilde = pi - alpha*angle_dx
                pj_tilde = pj + alpha*angle_dx

                pi_tilde[interaction_mask==2] = pi_tilde[interaction_mask==2] / torch.sqrt(torch.sum(pi_tilde[interaction_mask==2] ** 2, dim=1))[:, None]
                pj_tilde[interaction_mask==2] = pj_tilde[interaction_mask==2] / torch.sqrt(torch.sum(pj_tilde[interaction_mask==2] ** 2, dim=1))[:, None]
            else:
                pi_tilde = pi
                pj_tilde = pj


            S1 = torch.sum(torch.cross(pj_tilde, dx, dim=2) * torch.cross(pi_tilde, dx, dim=2), dim=2)            # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
            S2 = torch.sum(torch.cross(pi_tilde, qi, dim=2) * torch.cross(pj_tilde, qj, dim=2), dim=2)            # Calculating S2 (The ABP-PCP part of S).
            S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                        # Calculating S3 (The PCP-position part of S)

            if self.abs_s2s3:
                S2 = torch.abs(S2)
                S3 = torch.abs(S3)

            elif self.cell_wall_str != None:
                with torch.no_grad():
                    wall_mask = torch.sum(pi * pj , dim = 2) < 0.0
                    wall_mask = torch.logical_and(wall_mask , (interaction_mask == 2))
                    lam[wall_mask] = torch.tensor([self.cell_wall_str, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)


            if self.alpha != 0 and self.vesicle_formation:
                lam[:,:,3] = 0.0

            S = lam[:,:,0] + lam[:,:,1] * S1 + lam[:,:,2] * S2 + lam[:,:,3] * S3



        if (not(self.warming_up) and not(self.pre_polar)) and self.newnew_cellwall:
            wall_mask = torch.sum(pi * pj , dim = 2) < 0.0
            wall_mask = torch.logical_and(wall_mask , (interaction_mask == 2))
            Vij = z_mask[~wall_mask].float() * (torch.exp(-d[~wall_mask]) - S[~wall_mask] * torch.exp(-d[~wall_mask]/5))

            wall_potential = torch.where(d[wall_mask] < 2.011 , 1/2 * 10 * (2.011 - d[wall_mask])**2, 0.0)  #2.011 is the relaxation distance

            Vij_sum = torch.sum(Vij) + torch.sum(wall_potential)

        else:
            Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
            Vij_sum = torch.sum(Vij)

        if self.bound_radius:
            bc = self.sphere_bound(x)
        else:
            bc = 0.

        if (not self.warming_up and not self.pre_polar) and self.gamma:
            gauss_grad = self.gauss_grad(d, dx, interaction_mask)
            Vi     = torch.sum(self.gamma * p * gauss_grad, dim=1)
            return Vij_sum - torch.sum(Vi) + bc , int(m)
        else:
            return Vij_sum + bc, int(m)

    def init_simulation(self, x, p, q, p_mask):
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(p_mask) == len(x)

        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
        p_mask = torch.tensor(p_mask, dtype=torch.int, device=self.device)
        self.beta   = torch.zeros_like(p_mask, dtype=self.dtype, device=self.device)

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
        elif tstep < (self.pre_polar_dur + self.warmup_dur + self.vesicle_time) and self.alpha != 0:
            self.vesicle_formation = True
            self.seethru = 0  
            self.warming_up = False
            self.pre_polar = False
        elif self.alpha != 0:
            self.pre_polar  = False
            self.vesicle_formation = False
            self.warming_up = False
            self.tube_formation    = True
            self.alpha = self.tube_alpha
        else:
            self.seethru = 0
            self.warming_up = False
            self.pre_polar  = False

        if tstep == (self.warmup_dur + self.pre_polar_dur + self.prolif_delay + self.vesicle_time + 1):
            if self.polar_prolif_rate > 0:
                self.beta[p_mask == 1] = self.polar_prolif_rate
                self.beta[p_mask == 0] = self.non_polar_prolif_rate

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
        
        if not(self.warming_up) and not(self.pre_polar):

            with torch.no_grad():
                p[p_mask != 0] /= torch.sqrt(torch.sum(p[p_mask != 0] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
                q[p_mask != 0] /= torch.sqrt(torch.sum(q[p_mask != 0] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.

                p[p_mask == 0] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)
                q[p_mask == 0] = torch.tensor([0,0,0], device=self.device, dtype=self.dtype)

        # Calculate potential
        V, self.true_neighbour_max = self.potential(x, p, q, p_mask, idx, d, tstep)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x += -x.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            if not(self.warming_up) and not(self.pre_polar):
                p += -p.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
                q += -q.grad * self.dt + self.eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

                p.grad.zero_()
                q.grad.zero_()

        x.grad.zero_()

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
    
    def get_prolif_positions(self, p, q, p_mask):
        """
        Gives moves orthogonal to ABP of the dividing cells
        """
        with torch.no_grad():
            # We find all the polar particles
            p_polar = p[p_mask == 1].squeeze()
            q_polar = q[p_mask == 1].squeeze()

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

        beta = self.beta
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
                move[p_mask0 == 1] = self.get_prolif_positions(p0, q0, p_mask0)
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

    assert len(data_tuple) == 4 or len(data_tuple) == 2, 'data must be tuple of either len 2 (for data generation) or 4 (for data input)'
    

    np.random.seed(sim_dict['random_seed'])
    if len(data_tuple) == 4:
        p_mask, x, p, q = data_tuple
    else:
        data_gen = data_tuple[0]
        p_mask, x, p, q = data_gen(*data_tuple[1])
        
    sim = Simulation(sim_dict)
    runner = sim.simulation(x, p, q, p_mask)


    output_folder = sim_dict['output_folder']

    try:
        os.mkdir('data')
    except:
        pass
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
            save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)

    print(f'Simulation done, saved {yield_steps} datapoints')
    print('Took', time() - t1, 'seconds')

    save((p_mask_lst, x_lst, p_lst,  q_lst), name='data', output_folder=output_folder)