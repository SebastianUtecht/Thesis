from scipy.spatial import cKDTree
import numpy as np
import torch
import networkx as nx

### Setting device ###
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def find_potential_neighbours(x, k, distance_upper_bound=np.inf, workers=-1):
    tree = cKDTree(x)
    d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
    return d, idx

def find_true_neighbours(d, dx, seethru):
    with torch.no_grad():
        z_masks = []
        i0 = 0
        batch_size = 256
        i1 = batch_size
        while True:
            if i0 >= dx.shape[0]:
                break

            n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
            n_dis += 1000 * torch.eye(n_dis.shape[1], device=device)[None, :, :]

            z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= seethru
            z_masks.append(z_mask)

            if i1 > dx.shape[0]:
                break
            i0 = i1
            i1 += batch_size
    z_mask = torch.cat(z_masks, dim=0)
    return z_mask

def get_adj_lst(x, p_mask=None, seethru=0, k=10, p=None, p_threshold=0.0):
    d, idx = find_potential_neighbours(x, k=k)
    idx = torch.tensor(idx, dtype=torch.long, device=device)
    d = torch.tensor(d, dtype=torch.float, device=device)
    x = torch.tensor(x, dtype=torch.float, device=device)
    p_mask = torch.tensor(p_mask, dtype=torch.int, device=device)

    full_n_list = x[idx]
    dx = x[:, None, :] - full_n_list
    z_mask = find_true_neighbours(d, dx, seethru=seethru)

    sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)
    z_mask = torch.gather(z_mask, 1, sort_idx)
    idx = torch.gather(idx, 1, sort_idx)
    m = torch.max(torch.sum(z_mask, dim=1)) + 1
    z_mask = z_mask[:, :m]
    idx = idx[:, :m]

    if p is not None:
        pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
        pj = p[idx]
        wall_mask = (torch.sum(pi * pj , dim = 2) < 0.0) * (torch.sum(-dx * pj , dim = 2) < 0.0)
        z_mask[wall_mask] = 0

    polar_idx       = idx[p_mask == 1]
    return z_mask[p_mask == 1], polar_idx, p_mask #.detach().to("cpu").numpy()

def fix_polar_adj_arr(adj_arr, p_mask, z_mask):
    adj_arr[z_mask == 0] = -1
    polar_idx = np.argwhere(p_mask == 1).squeeze()
    adj_lst = [] 
    for sub_arr in adj_arr:
        polar_neighbors = list(np.intersect1d(sub_arr,polar_idx))
        adj_lst.append(polar_neighbors)
    return adj_lst

def make_edge_lst(adj_arr):
    edge_lst = []
    for sub_arr in adj_arr:
        for entry in sub_arr:
            edge_lst.append((sub_arr[0],entry))
    return edge_lst

def make_graph(edge_lst):
    G = nx.from_edgelist(edge_lst)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def get_edge_lst(polar_adj_arr, polar_z_mask, p_mask):
    polar_adj_arr[polar_z_mask == 0] = -1
    polar_idx = torch.argwhere(p_mask == 1).squeeze()
    non_polar_mask = torch.any(polar_adj_arr[:,:,None] == polar_idx, dim=2)
    double_mask = non_polar_mask * polar_z_mask

    masked_adj_arr = polar_adj_arr * double_mask
    edge_lst = torch.cat((polar_adj_arr[:,0][:,None].expand((polar_adj_arr.shape[0], polar_adj_arr.shape[1])).reshape(-1,1) , masked_adj_arr.reshape(-1,1)), dim=1) 
    edge_lst = edge_lst[edge_lst[:,1] != 0]

    return edge_lst

def get_percolation(x, p_mask, seethru=0):
    polar_z_mask, polar_adj_arr, p_mask = get_adj_lst(x, p_mask, seethru=seethru)
    n_polar_particles = polar_adj_arr.shape[0]
    edge_lst =  get_edge_lst(polar_adj_arr, p_mask = p_mask, polar_z_mask = polar_z_mask)
    graph = make_graph(edge_lst.detach().cpu().numpy())
    max_cluster_size = len(max(nx.connected_components(graph), key=len))
    perc_prob = max_cluster_size / n_polar_particles
    return perc_prob

def get_num_clusters(x, p_mask, threshold = 10, seethru=0, p=None):
    polar_z_mask, polar_adj_arr, p_mask = get_adj_lst(x, p_mask, seethru=seethru)
    edge_lst =  get_edge_lst(polar_adj_arr, p_mask = p_mask, polar_z_mask = polar_z_mask)
    graph = make_graph(edge_lst.detach().cpu().numpy())
    lst = []
    for comp in nx.connected_components(graph):
        lst.append(len(comp) > threshold)
    return sum(lst)

def get_branching(x, q, p_mask, threshold=0.1, seethru=0):
    _, idx = adj_lst(x, p_mask, seethru=seethru)
    idx = idx[:,1:]

    p_mask = torch.tensor(p_mask, dtype=torch.int, device=device)
    idx = torch.tensor(idx, device=device)
    q = torch.tensor(q, device=device)
    q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]

    q[p_mask == 0] = torch.tensor([0.,0.,0.], device=device)

    qj = q[idx]
    qi = q[p_mask == 1][:, None, :].expand(q[p_mask == 1].shape[0], idx.shape[1], 3)

    sine_vals = torch.sum(torch.abs(qi * qj), dim=2)
    
    branch_connections = sine_vals < threshold
    branch_connections[sine_vals == 0] = 0
    branch_particles = idx[branch_connections]

    return branch_particles.detach().to("cpu").numpy()

def course_graining(x, p_mask, levels=1, k=5):
    first = True

    for _ in range(levels):
        if first:
            z_mask, idx = adj_lst(x, p_mask, k=k)
            idx = fix_polar_adj_arr(idx, p_mask, z_mask)
            first = False
        else:
            _, idx = find_potential_neighbours(x, k=k)

        coarse_pos_lst  = []
        used_idx_lst = []

        for i in range(len(idx)):
            used_idx = [x for x in idx[i] if x not in used_idx_lst]
            if not used_idx:
                continue

            coarse_pos = x[used_idx].mean(axis=0)
            coarse_pos_lst.append(coarse_pos)
            used_idx_lst += used_idx
        
        x = np.array(coarse_pos_lst)
    return x

def find_sphere_neighbours(x, cutoff):
    all_dists = torch.sum((x[ :, None, :] - x[ None, :, :]) ** 2, dim=2)
    dists, indices = torch.sort(all_dists)
    # dists, indices = dists[:, 1:], indices[:,1:]

    z_mask = torch.zeros_like(dists, device=device, dtype=torch.int)
    z_mask[ dists < cutoff**2 ] = 1
    m = torch.max(torch.sum(z_mask, dim=1)) + 1

    dists, indices, z_mask = dists[:,:m], indices[:,:m], z_mask[:,:m]
    return torch.sqrt(dists), indices, z_mask

def adj_lst_sphere(x, p_mask, radius=5):
    x = torch.tensor(x, dtype=torch.float, device=device)
    _, idx, z_mask = find_sphere_neighbours(x, radius)
    
    z_mask = z_mask.detach().to("cpu").numpy()
    x = x.detach().to("cpu").numpy()
    idx = idx.detach().to("cpu").numpy()
    idx = idx[p_mask == 1]
    z_mask = z_mask[p_mask == 1]
    return z_mask, idx

def sphere_coarse_graining(x, p_mask, radius):
    x = torch.tensor(x, dtype=torch.float, device=device)
    _, idx, z_mask = find_sphere_neighbours(x, radius)
    
    z_mask = z_mask.detach().to("cpu").numpy()
    x = x.detach().to("cpu").numpy()
    idx = idx.detach().to("cpu").numpy()
    idx = idx[p_mask == 1]
    idx = fix_polar_adj_arr(idx, p_mask, z_mask)
    
    coarse_pos_lst  = []
    used_idx_lst = []

    for i in range(len(idx)):
        used_idx = [x for x in idx[i] if x not in used_idx_lst]
        if not used_idx:
            continue

        coarse_pos = x[used_idx].mean(axis=0)
        coarse_pos_lst.append(coarse_pos)
        used_idx_lst += used_idx
    
    x = np.array(coarse_pos_lst)
    return x

def get_centers(x, p, q, p_mask, depression=2  ,num=500):    
    x_polar = x[p_mask == 1].copy()
    p_polar = p[p_mask == 1].copy()
    q_polar = q[p_mask == 1].copy()

    p_polar /= np.sqrt(np.sum(p_polar ** 2, axis=1))[:, None]
    q_polar -= np.sum(q_polar * p_polar, axis=1)[:,None] * p_polar
    q_polar /= np.sqrt(np.sum(q_polar ** 2, axis=0))

    lin_trans = np.zeros((len(x_polar), 3, 3))
    lin_trans[:,:,0] = p_polar
    lin_trans[:,:,1] = q_polar
    lin_trans[:,:,2] = np.cross(p_polar,q_polar)

    x_trans = (np.linalg.inv(lin_trans) @ x_polar[:,:,None]).squeeze()

    subtract_arr =  np.zeros_like(x_trans)
    subtract_arr[:,0] = -depression

    trans_centers =  x_trans - subtract_arr
    centers = (lin_trans @ trans_centers[:,:,None]).squeeze()
    sampled_centers = centers[:num]

    return sampled_centers

def above_threshold(x, p_mask, max_fraction=0.8 , radius=None, k=100):
    if not radius:
        z_mask, idx = adj_lst(x, p_mask, k=k)
        idx = fix_polar_adj_arr(idx, p_mask, z_mask)
        len_lst = [len(x) for x in idx]
        max_neighbors = max(len_lst)
        threshold = max_neighbors * max_fraction
        mask = [neighbors > threshold for neighbors in len_lst]
        return_lst = []
        for i in range(len(mask)):
            if mask[i]:
                return_lst.append(idx[i][0])
    
    else:
        x = torch.tensor(x, dtype=torch.float, device=device)
        _, idx, z_mask = find_sphere_neighbours(x, radius)
        
        z_mask = z_mask.detach().to("cpu").numpy()
        x = x.detach().to("cpu").numpy()
        idx = idx.detach().to("cpu").numpy()
        idx = idx[p_mask == 1]
        idx = fix_polar_adj_arr(idx, p_mask, z_mask)

        len_lst = [len(x) for x in idx]
        max_neighbors = max(len_lst)
        threshold = int(max_neighbors * max_fraction)
        print(threshold)
        mask = [neighbors > threshold for neighbors in len_lst]
        return_lst = []
        for i in range(len(mask)):
            if mask[i]:
                return_lst.append(idx[i][0])

    return return_lst


