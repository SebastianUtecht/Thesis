import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
from vispy import app

def export_png(folder, timestep, name, alpha=10, view_particles=None):

    # Getting the data
    data = np.load(folder, allow_pickle=True)
    mask_lst, x_lst, p_lst, q_lst = data

    for p_mask in mask_lst:
        p_mask[p_mask == 2] = 1

    p_lst = [p_lst[i]/ np.sqrt(np.sum(p_lst[i] ** 2, axis=1))[:, None] for i in range(len(p_lst))]

    polar_pos_lst = []
    polar_pos_lst = [x_lst[i][mask_lst[i] == 1] + 0.2 * p_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(bgcolor='white')
    view = canvas.central_widget.add_view()    
    view.camera = "arcball"            
    view.camera.distance = 150                   

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True, light_ambient=0.1, light_color='white')
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True, light_ambient=0.1, light_color='white')
    scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True, light_ambient=0.1, light_color='white')

    scatter1.set_data(x_lst[timestep][mask_lst[timestep] == 0] , edge_width=0, face_color='blue', size=2.5)
    scatter2.set_data(x_lst[timestep][mask_lst[timestep] == 1], edge_width=0, face_color='red', size=2.5)
    scatter3.set_data(polar_pos_lst[timestep] , edge_width=0, face_color='red', size=2.5)

    # Add the scatter object to the view
    if not view_particles:
        view.add(scatter1)
        view.add(scatter2)
        # view.add(scatter3)
    else:
        assert view_particles == "polar" or view_particles == "non_polar", "view_particles only takes arguments polar or non_polar"
        if view_particles == 'polar':
            view.add(scatter2)
            # view.add(scatter3)
        if view_particles == 'non_polar':
            view.add(scatter1)


    # Use render to generate an image object
    img=canvas.render()

    # Use write_png to export your wonderful plot as png ! 
    vispy.io.write_png(name + ".png",img)

    return None



start_folder= "data/vitro_gradves_grid/"
conc_lst    = np.linspace(0.5, 0.99, 15)
conc = conc_lst[11-9]
print(1-conc)

folder = start_folder + f'conc_{conc:.2f}/data.npy'
export_png(folder, timestep=33 , name=f'plots/gradves_conc_{1-conc:.2f}_tot', view_particles=None)