import numpy as np
import vispy.scene
import vispy
from vispy.scene import visuals
import sys
from vispy import app
from vispy import scene
from vispy.visuals.transforms import STTransform

def dissect_sphere(x, partitions=6, dir=None):

    color_lst = ['white', 'red', 'blue', 'green', 'yellow', 'purple']

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()         

    if np.any(dir):

        arrows = vispy.scene.visuals.Line(width=10, color='blue')
        arrows.set_data(pos=np.array([[0,0,0],dir*30]))
        view.add(arrows)

        def transfer_matrix(x_trans_vec):
            x_trans_vec /= np.sqrt(np.sum(x_trans_vec ** 2))
            nx, ny, nz = x_trans_vec
            trans_mat = np.array([[nx, -nx*ny/(1+nz), 1-(nx**2/(1+nz))],
                                    [ny, 1-(ny**2/(1+nz)), -nx*ny/(1+nz)],
                                    [nz,        -ny,          -nx]])
            return trans_mat
    
        trans_mat       = transfer_matrix(dir)
        trans_mat_inv   = np.linalg.inv(trans_mat)
        trans_mat_inv_expanded = trans_mat_inv[None,:,:].repeat(len(x),axis=0)
        transformed_x   = (trans_mat_inv_expanded @ x[:,:,None]).squeeze()
    else:
        transformed_x = x

    part_lines = np.linspace(np.min(transformed_x[:,0])-0.1, np.max(transformed_x[:,0])+0.1, partitions + 1)

    partition_mask = np.zeros(len(transformed_x))
    for i in range(partitions-1):
        cells_in_partition = np.argwhere((transformed_x[:,0] > part_lines[i+1]) * (transformed_x[:,0] < part_lines[i+2]))
        partition_mask[cells_in_partition] = i+1

    for cell_type in np.unique(partition_mask):
        cell_type = int(cell_type)
        exec(f'scatter{cell_type} = visuals.Markers(scaling=True, alpha=10, spherical=True)')
        exec(f'scatter{cell_type}.set_data(x[partition_mask == cell_type] , edge_width=0, face_color=color_lst[cell_type], size=2.5)')
        exec(f'view.add(scatter{cell_type})')
    
    view.camera = 'fly'
    
    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()
    
    return None

folder = "data/test_sphere/data.npy"
data = np.load(folder, allow_pickle=True)
x = data[1][-1]

dissect_sphere(x, dir=np.array([10.0,1.0,-20.0]))