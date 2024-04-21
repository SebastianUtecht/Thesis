import numpy as np
import vispy.scene
import vispy
from vispy.scene import visuals
import sys
from vispy import app


iterator = 0
def interactive_animate(folder, alpha=10, view_particles=None, interval=1/60):
    
    # Getting the data
    data = np.load(folder, allow_pickle=True)
    mask_lst, x_lst, p_lst, q_lst = data
    p_lst = [p_lst[i]/ np.sqrt(np.sum(p_lst[i] ** 2, axis=1))[:, None] for i in range(len(p_lst))]

    polar_pos_lst = []
    polar_pos_lst = [x_lst[i][mask_lst[i] == 1] + 0.2 * p_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                                   

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)

    if len(x_lst[0][mask_lst[0] == 0]) == 0:
        np_data = np.array([[0,0,0],[0,0,0]])
    else:
        np_data = x_lst[0][mask_lst[0] == 0]
    scatter1.set_data(np_data , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
    scatter2.set_data(x_lst[0][mask_lst[0] == 1], edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
    scatter3.set_data(polar_pos_lst[0] , edge_width=0, face_color='red', size=2.5)

    # Add the scatter object to the view
    if not view_particles:
        view.add(scatter1)
        view.add(scatter2)
        view.add(scatter3)
    else:
        assert view_particles == "polar" or view_particles == "non_polar", "view_particles only takes arguments polar or non_polar"
        if view_particles == 'polar':
            view.add(scatter2)
            view.add(scatter3)
        if view_particles == 'non_polar':
            view.add(scatter1)


    def update(ev):
        global x, iterator
        iterator += 1
        x = x_lst[int(iterator) % len(x_lst)]
        mask = mask_lst[int(iterator) % len(x_lst)]
        polar_pos = polar_pos_lst[int(iterator) % len(x_lst)]
        if len(x[mask == 0]) == 0:
            np_data = np.array([[0,0,0],[0,0,0]])
        else:
            np_data = x[mask == 0]
        scatter1.set_data(np_data , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
        scatter2.set_data(x[mask == 1], edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
        scatter3.set_data(polar_pos , edge_width=0, face_color='red', size=2.5)


    timer = app.Timer(interval=interval)
    timer.connect(update)
    timer.start()

    @canvas.connect
    def on_key_press(event):
        global iterator
        if event.text == ' ':
            if timer.running:
                timer.stop()
            else:
                timer.start()
        elif event.text == 'r':
            iterator -= 51
            update(1)
        elif event.text == 't':
            iterator += 49
            update(1)
        elif event.text == ',':
            iterator -= 2
            update(1)
        elif event.text == '.':
            update(1)

    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()

iterator = 0
def animate_in_vivo(folder, alpha=10, view_particles=None, interval=1/60):
    
    # Getting the data
    data = np.load(folder, allow_pickle=True)
    mask_lst, x_lst, p_lst, q_lst = data
    p_lst = [p_lst[i]/ np.sqrt(np.sum(p_lst[i] ** 2, axis=1))[:, None] for i in range(len(p_lst))]

    polar_pos_lst  = [x_lst[i][mask_lst[i] == 1] + 0.2 * p_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]
    polar_pos_lst += [x_lst[i][mask_lst[i] == 3] + 0.2 * p_lst[i][mask_lst[i] == 3] for i in range(len(x_lst))]
    polar_pos_lst += [x_lst[i][mask_lst[i] == 4] + 0.2 * p_lst[i][mask_lst[i] == 4] for i in range(len(x_lst))]

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                                   

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter4 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter5 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter6 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter7 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)

    scatter1.set_data(x_lst[0][mask_lst[0] == 1], edge_width=0, face_color='blue', size=2.5)
    scatter2.set_data(x_lst[0][mask_lst[0] == 0], edge_width=0, face_color='white', size=2.5)
    scatter3.set_data(polar_pos_lst[0] , edge_width=0, face_color='red', size=2.5)

    # Add the scatter object to the view
    if not view_particles:
        view.add(scatter1)
        view.add(scatter2)
        view.add(scatter3)
    else:
        assert view_particles == "polar" or view_particles == "non_polar", "view_particles only takes arguments polar or non_polar"
        if view_particles == 'polar':
            view.add(scatter2)
            view.add(scatter3)
        if view_particles == 'non_polar':
            view.add(scatter1)


    def update(ev):
        global x, iterator
        iterator += 1
        x = x_lst[int(iterator) % len(x_lst)]
        mask = mask_lst[int(iterator) % len(x_lst)]
        polar_pos = polar_pos_lst[int(iterator) % len(x_lst)]
        if np.sum(mask == 0) > 0:
            scatter1.set_data(x[mask == 0], edge_width=0, face_color='blue', size=2.5)
        if np.sum(mask == 1) > 0:
            scatter2.set_data(x[mask == 1], edge_width=0, face_color='white', size=2.5)
            scatter3.set_data(polar_pos , edge_width=0, face_color='red', size=2.5)
        if np.sum(mask == 2) > 0:
            scatter4.set_data(x[mask == 2], edge_width=0, face_color='green', size=2.5)
            view.add(scatter4)
            view.add(scatter5)
        if np.sum(mask == 3) > 0:
            scatter5.set_data(x[mask == 3], edge_width=0, face_color='orange', size=2.5)
            view.add(scatter5)
        if np.sum(mask == 4) > 0:
            scatter5.set_data(x[mask == 4], edge_width=0, face_color='yellow', size=2.5)
            view.add(scatter6)
        if np.sum(mask == 5) > 0:
            scatter7.set_data(x[mask == 5], edge_width=0, face_color='purple', size=2.5)
            view.add(scatter7)

        
    timer = app.Timer(interval=interval)
    timer.connect(update)
    timer.start()

    @canvas.connect
    def on_key_press(event):
        global iterator
        if event.text == ' ':
            if timer.running:
                timer.stop()
            else:
                timer.start()
        elif event.text == 'r':
            iterator -= 51
            update(1)
        elif event.text == 't':
            iterator += 49
            update(1)
        elif event.text == ',':
            iterator -= 2
            update(1)
        elif event.text == '.':
            update(1)

    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()


def interactive_plot(folder, timestep, alpha=10, view_particles=None):

    # Getting the data
    data = np.load(folder, allow_pickle=True)
    mask_lst, x_lst, p_lst, q_lst = data
    p_lst = [p_lst[i]/ np.sqrt(np.sum(p_lst[i] ** 2, axis=1))[:, None] for i in range(len(p_lst))]
    q_lst = [q_lst[i]/ np.sqrt(np.sum(q_lst[i] ** 2, axis=1))[:, None] for i in range(len(q_lst))]

    polar_pos_lst = [x_lst[i][mask_lst[i] == 1] + 0.2 * p_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]
    pcp_pos_lst   = [x_lst[i][mask_lst[i] == 1] + 1.5 * p_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]
    pcp_dir_lst   = [pcp_pos_lst[i] + q_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]
    pcp_plot_lst  = [np.concatenate((pcp_dir_lst[i] , pcp_pos_lst[i]), axis=0) for i in range(len(x_lst))]

    pcp_plot_connections = [np.concatenate( (np.arange(np.sum(mask_lst[i] == 1))[:,None], np.arange(np.sum(mask_lst[i] == 1))[:,None] + np.sum(mask_lst[i] == 1)), axis=1 ) for i in range(len(x_lst))]

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                                   

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter4 =  vispy.scene.visuals.Line(width=5, color='blue')

    scatter1.set_data(x_lst[timestep][mask_lst[timestep] == 0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
    scatter2.set_data(x_lst[timestep][mask_lst[timestep] == 1], edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
    scatter3.set_data(polar_pos_lst[timestep] , edge_width=0, face_color='red', size=2.5)
    scatter4.set_data(pos=pcp_plot_lst[timestep], connect=pcp_plot_connections[timestep])

    # Add the scatter object to the view
    if not view_particles:
        view.add(scatter1)
        view.add(scatter2)
        view.add(scatter3)
        view.add(scatter4)
    else:
        assert view_particles == "polar" or view_particles == "non_polar", "view_particles only takes arguments polar or non_polar"
        if view_particles == 'polar':
            view.add(scatter2)
            view.add(scatter3)
            view.add(scatter4)
        if view_particles == 'non_polar':
            view.add(scatter1)


    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()

def visualize_neighbors(x, p_mask, idx, alpha=10):

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                                   

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter3 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)

    scatter1.set_data(x[p_mask == 1] , edge_width=0, face_color='white', size=2.5)
    scatter2.set_data(np.array([x[idx[0][0]]]), edge_width=0, face_color='red', size=3)
    scatter3.set_data(x[idx[0][1:]] , edge_width=0, face_color='blue', size=3)

    view.add(scatter1)
    view.add(scatter2)
    view.add(scatter3)

    @canvas.connect
    def on_key_press(event):
        global iterator
        if event.text == 'r':
            iterator -= 1
            iterator =  iterator % np.sum(p_mask == 1)
            scatter2.set_data(np.array([x[idx[iterator][0]]]), edge_width=0, face_color='red', size=3)
            scatter3.set_data(x[idx[iterator][1:]] , edge_width=0, face_color='blue', size=3)
        elif event.text == 't':
            iterator += 1
            iterator =  iterator % np.sum(p_mask == 1)
            scatter2.set_data(np.array([x[idx[iterator][0]]]), edge_width=0, face_color='red', size=3)
            scatter3.set_data(x[idx[iterator][1:]] , edge_width=0, face_color='blue', size=3)
            
    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()

def display_branches(folder, timestep, branch_idx, alpha=10):

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    # Getting the data
    data = np.load(folder, allow_pickle=True)
    mask, x_lst, _, _ = data

    mask = mask[timestep]
    x = x_lst[timestep]
    x = x

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter1.set_data(x[mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)

    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2.set_data(x[branch_idx] , edge_width=0, face_color='red', size=2.7)

    view.add(scatter1)
    view.add(scatter2)

    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()