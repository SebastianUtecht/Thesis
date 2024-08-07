from Animate import *
from Metrics import *
from DataGeneration import *
import os
from vispy import scene
from vispy.visuals.transforms import STTransform
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def data_plot(mask, x, p, q):

    polar_pos_lst = x[mask == 1] + 0.2 * p[mask == 1]
    pcp_pos_lst   = x[mask == 1] + 1.5 * p[mask == 1]
    pcp_dir_lst   = pcp_pos_lst + q[mask == 1]
    pcp_plot_lst  = np.concatenate((pcp_dir_lst , pcp_pos_lst), axis=0)

    pcp_plot_connections = np.concatenate( (np.arange(np.sum(mask == 1))[:,None], np.arange(np.sum(mask == 1))[:,None] + np.sum(mask == 1)), axis=1 )

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                                   

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=10, spherical=True)
    scatter3 = visuals.Markers(scaling=True, alpha=10, spherical=True)
    scatter2 = visuals.Markers(scaling=True, alpha=10, spherical=True)
    # scatter4 =  vispy.scene.visuals.Line(width=5, color='blue')

    scatter1.set_data(x[mask == 0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
    scatter2.set_data(x[mask == 1], edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
    scatter3.set_data(polar_pos_lst , edge_width=0, face_color='red', size=2.5)
    # scatter4.set_data(pos=pcp_plot_lst, connect=pcp_plot_connections)

    # Add the scatter object to the view
    if x[mask == 0].shape[0] > 0:
        view.add(scatter1)
    view.add(scatter2)
    view.add(scatter3)
    # view.add(scatter4)

    sphere1 = scene.visuals.Sphere(radius=20, method='latitude', parent=view.scene,
                               edge_color='blue')

    sphere1.transform = STTransform(translate=[0, 0, 5])

    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()

# data = np.load('data/ubud_mcap_final.npy', allow_pickle=True)

# data = (data[0][-1], data[1][-1], data[2][-1], data[3][-1])
# data[1][:,-1] -= 37.5
# data[1][:,1] += 17.5




data_plot(*data)

# def save(data_tuple, name, output_folder):
#     with open(f'{output_folder}/{name}.npy', 'wb') as f:
#         pickle.dump(data_tuple, f)

# save(data, 'ubud_translated', 'data')




