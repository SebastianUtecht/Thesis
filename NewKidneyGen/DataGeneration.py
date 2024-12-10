import numpy as np
import os
import pickle
from sklearn.datasets import make_blobs

def make_random_cube(N, non_polar_frac, size=20):

    """ Returns a numpy array with random positions (withing size x size cube),
        APB and PCP.
        Along with a mask dictating which particles are non polarized """
    
    x = np.random.uniform(-size/2,size/2, size=(N,3))                                   #Particles positions

    p = np.random.uniform(-1,1,size=(N,3))                                              #AB polarity unit vectors
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]

    q = np.random.uniform(-1,1,size=(N,3))                                              #PCP unit vectors
    q /= np.sqrt(np.sum(p**2, axis=1))[:,None]

    mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)        #Mask detailing which particles are non polar
    p[mask == 0] = np.array([0,0,0])                                                    #Setting the polarities of the non-polarized particles to 0
    q[mask == 0] = np.array([0,0,0])

    cube_data = (mask, x, p, q)                                                #Total data
    return cube_data

# def make_random_sphere(N, non_polar_frac, radius=20):

#     """ Returns a numpy array with random positions (withing size x size cube),
#         APB and PCP.
#         Along with a mask dictating which particles are non polarized """
    
#     cube_data = make_random_cube(N*5, non_polar_frac, radius*2)
#     sphere_mask = (cube_data[1]**2).sum(axis=1) <= radius**2
#     sphere_data = (cube_data[0][sphere_mask][:N],
#                 cube_data[1][sphere_mask][:N],
#                 cube_data[2][sphere_mask][:N],
#                 cube_data[3][sphere_mask][:N])
    
#     if sphere_data[0].size != N:
#         make_random_sphere(N, non_polar_frac, radius)

#     else:
#         return sphere_data

def init_cylinder(n, radius=10, length=50):
    phi = 2 * np.pi * np.random.random(n)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = length * np.random.random(n)

    x = np.array([x, y, z]).T

    p = x.copy()
    p -= np.mean(p, axis=0)
    p[:, 2] = 0
    q = np.cross(p, [0, 0, 1])

    return x, p, q

def make_disc(N, radius=10):
    x = np.random.randn(N, 3)
    x[:,-1] = 0.0
    r = radius * np.random.rand(N)**(1/2.)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= r[:, None]

    p = np.repeat(np.array([[0.0,0.0,1.0]]),N, axis=0)
    q = np.cross(x,p)
    q /= np.sqrt(np.sum(q**2, axis=1))[:,None]

    mask = np.ones(N)

    disc_data = (mask, x, p, q)
    return disc_data

def make_cylinder_cross(n_v, n_h, n_c, v_len, h_len, radius=7):

    x_up, p_up, q_up        = init_cylinder(n_v, radius, v_len)
    x_flat, p_flat, q_flat  = init_cylinder(n_h, radius, h_len)

    rot_mat = np.array([[1.0,0.0,0.0],[0.0,0.0, -1],[0,1, 0]])
    rot_mat = np.repeat(rot_mat[None], n_h, axis=0)

    x_flat = (rot_mat @ x_flat[:,:,None]).squeeze()
    p_flat = (rot_mat @ p_flat[:,:,None]).squeeze()
    q_flat = (rot_mat @ q_flat[:,:,None]).squeeze()

    _, disc1, disc_p, disc_q = make_disc(N = n_c, radius=radius)

    disc1 = (rot_mat[: n_c] @ disc1[:,:,None]).squeeze()
    disc1[:,2] += (v_len + radius/2) 
    disc2 = disc1.copy()
    disc1[:,1] += h_len/2
    disc2[:,1] -= h_len/2
    
    disc_p = (rot_mat[: n_c] @ disc_p[:,:,None]).squeeze()
    disc_q = (rot_mat[: n_c] @ disc_q[:,:,None]).squeeze()
    
    disc2_p = + disc_p.copy()
    disc1_p = - disc_p.copy()

    x_flat[:,1] += h_len/2
    x_flat[:,2] += (v_len + radius/2) 

    x = np.concatenate((x_up, x_flat, disc1, disc2), axis=0)
    p = np.concatenate((p_up, p_flat, disc1_p, disc2_p), axis=0)
    q = np.concatenate((q_up, q_flat, disc_q, disc_q), axis=0)
    mask = np.ones(len(x))
    return (mask, x, p, q)

def make_random_sphere(N, non_polar_frac , radius=35):
    x = np.random.randn(N, 3)
    r = radius * np.random.rand(N)**(1/3.)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    x *= r[:, None]

    p = np.random.randn(N, 3)
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]
    q = np.random.randn(N, 3)
    q /= np.sqrt(np.sum(p**2, axis=1))[:,None]

    mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)        #Mask detailing which particles are non polar
    p[mask == 0] = np.array([0,0,0])                                                    #Setting the polarities of the non-polarized particles to 0
    q[mask == 0] = np.array([0,0,0])

    sphere_data = (mask, x, p, q)
    return sphere_data

def make_2_spheres(N1, N2, N_np=0, x_trans=20, y_trans=0, z_trans=0,  radius=1):

    p_mask1, x1, p1, q1 = make_random_sphere(N1, 0, radius=radius)
    p_mask2, x2, p2, q2 = make_random_sphere(N2, 0, radius=radius)

    x2 = x2 + np.array([x_trans, y_trans, 0])

    x       = np.concatenate((x1, x2), axis=0)
    p_mask  = np.concatenate((p_mask1, p_mask2), axis=0) 
    p = np.concatenate((p1, p2), axis=0)
    q = np.concatenate((q1, q2), axis=0)
    
    if N_np:
        p_mask_np, x_np, p_np, q_np = make_random_sphere(N_np, 1, radius=x_trans*0.8)
        x_np = x_np + np.array([x_trans/2, 0, 0])
        
        p_mask = np.concatenate((p_mask, p_mask_np), axis=0)
        x = np.concatenate((x, x_np), axis=0)
        p = np.concatenate((p, p_np), axis=0)
        q = np.concatenate((q, q_np), axis=0)

    data = (p_mask, x, p, q)

    return data

def make_covered_blob(N_polar, N_non, non_polar_radius=35):
    polar_blob      = make_random_sphere(N_polar, non_polar_frac=0.0, radius=0.1)
    non_polar_blob  = make_random_sphere(N_non, non_polar_frac=1.0, radius=non_polar_radius)

    tot_mask = np.concatenate((polar_blob[0], non_polar_blob[0]))
    tot_x    = np.concatenate((polar_blob[1], non_polar_blob[1]))
    tot_p    = np.concatenate((polar_blob[2], non_polar_blob[2]))
    tot_q    = np.concatenate((polar_blob[3], non_polar_blob[3]))

    return (tot_mask, tot_x, tot_p, tot_q)

def make_n_save(N, non_polar_frac, function, size, name, folder='data'):
    data = function(N, non_polar_frac, size)

    try:
        os.mkdir(folder)
    except:
        pass

    with open(folder + '/' + name + '.npy', 'wb') as f:
        pickle.dump(data, f)

# def make_n_save_cube(N, non_polar_frac, size, name):                                        #We save the data
#     cube_data = make_random_cube(N, non_polar_frac, size)
#     np.save(name,cube_data)
#     return None

# def make_random_blobs(N, N_blobs, non_polar_frac, blob_std=1 , size=20, rand_seed=None):

#     """ Returns a numpy array with random positions that are generated 
#         within a size x size cube as gaussian blobs, APB and PCP.
#         Along with a mask dictating which particles are non polarized """
    
#     x, _ = make_blobs(n_samples=N,                                                    
#                                       n_features=3,
#                                       centers=N_blobs,
#                                       cluster_std=blob_std, 
#                                       center_box=(-size/2, size/2),
#                                       random_state=rand_seed)                          #Particle positions

#     p = np.random.uniform(-1,1,size=(N,3))                                             #AB polarity unit vectors
#     p /= np.sqrt(np.sum(p**2, axis=1))[:,None]

#     q = np.random.uniform(-1,1,size=(N,3))                                             #PCP unit vectors
#     q /= np.sqrt(np.sum(p**2, axis=1))[:,None]

#     mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)       #Mask detailing which particles are non polar
#     p[mask == 0] = np.array([0,0,0])                                                   #Setting the polarities of the non-polarized particles to 0
#     q[mask == 0] = np.array([0,0,0])

#     blobs_data = np.concatenate((mask[:,None], x, p, q) ,axis=1)                       #Total data
#     return blobs_data

# def make_n_save_blobs(N, N_blobs, non_polar_frac, size, name, 
#                       rand_seed=None, blob_std=1):                                     #We save the data
    
#     blobs_data = make_random_blobs(N=N, N_blobs=N_blobs, none_polar_frac=non_polar_frac,
#                                     blob_std=blob_std, size=size, rand_seed=rand_seed)
#     np.save(name,blobs_data)
    
#     return None

