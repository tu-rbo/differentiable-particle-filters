import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

head_scale = 3.0  # 1.5
# head_scale = 2.0  # 1.5
quiv_kwargs = {'scale_units':'xy', 'scale':1./80., 'width': 0.003, 'headlength': 5*head_scale, 'headwidth': 3*head_scale, 'headaxislength': 4.5*head_scale}


def show_pause(show=False, pause=0.0):
    '''Shows a plot by either blocking permanently using show or temporarily using pause.'''
    if show:
        plt.ioff()
        plt.show()
    elif pause:
        plt.ion()
        plt.pause(pause)


def plot_maze(maze='nav01', margin=1, means=None, stds=None, figure_name=None, show=False, pause=False, ax=None, linewidth=1.0):
    if ax is None:
        ax = plt.gca()
    if figure_name is not None:
        plt.figure(figure_name)

    if 'nav01' in maze:
        walls = np.array([
            # horizontal
            [[0, 500], [1000, 500]],
            [[400, 400], [500, 400]],
            [[600, 400], [700, 400]],
            [[800, 400], [1000, 400]],
            [[200, 300], [400, 300]],
            [[100, 200], [200, 200]],
            [[400, 200], [700, 200]],
            [[200, 100], [300, 100]],
            [[600, 100], [900, 100]],
            [[0, 0], [1000, 0]],
            # vertical
            [[0, 0], [0, 500]],
            [[100, 100], [100, 200]],
            [[100, 300], [100, 500]],
            [[200, 200], [200, 400]],
            [[200, 0], [200, 100]],
            [[300, 100], [300, 200]],
            [[300, 400], [300, 500]],
            [[400, 100], [400, 400]],
            [[500, 0], [500, 200]],
            [[600, 100], [600, 200]],
            [[700, 200], [700, 300]],
            [[800, 200], [800, 400]],
            [[900, 100], [900, 300]],
            [[1000, 0], [1000, 500]],
        ])
        rooms = [
            # [[400, 200], 300, 200]
            ]
        ax.set_xlim([-margin, 1000+margin])
        ax.set_ylim([-margin, 500+margin])

    if 'nav02' in maze:
        walls = np.array([
            # horizontal
            [[0, 900], [1500, 900]],
            [[100, 800], [400, 800]],
            [[500, 800], [600, 800]],
            [[800, 800], [1000, 800]],
            [[1100, 800], [1200, 800]],
            [[1300, 800], [1400, 800]],
            [[100, 700], [600, 700]],
            [[700, 700], [800, 700]],
            [[1000, 700], [1100, 700]],
            [[1200, 700], [1400, 700]],
            [[900, 600], [1200, 600]],
            [[1300, 600], [1500, 600]],
            [[0, 500], [100, 500]],
            [[1300, 500], [1400, 500]],
            [[100, 400], [200, 400]],
            [[1200, 400], [1400, 400]],
            [[300, 300], [800, 300]],
            [[900, 300], [1200, 300]],
            [[400, 200], [600, 200]],
            [[700, 200], [800, 200]],
            [[1200, 200], [1500, 200]],
            [[200, 100], [300, 100]],
            [[500, 100], [700, 100]],
            [[800, 100], [900, 100]],
            [[1100, 100], [1400, 100]],
            [[0, 0], [1500, 0]],
            # vertical
            [[0, 0], [0, 900]],
            [[100, 0], [100, 300]],
            [[100, 500], [100, 600]],
            [[100, 700], [100, 800]],
            [[200, 100], [200, 200]],
            [[200, 300], [200, 400]],
            [[200, 500], [200, 700]],
            [[300, 100], [300, 300]],
            [[400, 0], [400, 200]],
            [[500, 800], [500, 900]],
            [[700, 100], [700, 200]],
            [[700, 700], [700, 800]],
            [[800, 200], [800, 800]],
            [[900, 100], [900, 700]],
            [[1000, 0], [1000, 200]],
            [[1000, 700], [1000, 800]],
            [[1100, 700], [1100, 800]],
            [[1100, 100], [1100, 300]],
            [[1200, 800], [1200, 900]],
            [[1200, 400], [1200, 700]],
            [[1300, 200], [1300, 300]],
            [[1300, 500], [1300, 600]],
            [[1400, 300], [1400, 500]],
            [[1400, 700], [1400, 800]],
            [[1500, 0], [1500, 900]],
        ])
        rooms = [
            # [[900, 300], 300, 300]
            ]
        ax.set_xlim([-margin, 1500+margin])
        ax.set_ylim([-margin, 900+margin])

    if 'nav03' in maze:
        walls = np.array([
            # horizontal
            [[0, 1300], [2000, 1300]],
            [[100, 1200], [500, 1200]],
            [[600, 1200], [1400, 1200]],
            [[1600, 1200], [1700, 1200]],
            [[0, 1100], [600, 1100]],
            [[1500, 1100], [1600, 1100]],
            [[1600, 1000], [1800, 1000]],
            [[800, 1000], [900, 1000]],
            [[100, 1000], [200, 1000]],
            [[700, 900], [800, 900]],
            [[1600, 900], [1800, 900]],
            [[200, 800], [300, 800]],
            [[800, 800], [1200, 800]],
            [[1300, 800], [1500, 800]],
            [[1600, 800], [1900, 800]],
            [[900, 700], [1400, 700]],
            [[1500, 700], [1600, 700]],
            [[1700, 700], [1900, 700]],
            [[700, 600], [800, 600]],
            [[1400, 600], [1500, 600]],
            [[1600, 600], [1700, 600]],
            [[100, 500], [200, 500]],
            [[300, 500], [500, 500]],
            [[600, 500], [700, 500]],
            [[1400, 500], [1900, 500]],
            [[100, 400], [200, 400]],
            [[400, 400], [600, 400]],
            [[1500, 400], [1600, 400]],
            [[1700, 400], [1800, 400]],
            [[200, 300], [300, 300]],
            [[400, 300], [500, 300]],
            [[600, 300], [800, 300]],
            [[900, 300], [1100, 300]],
            [[1300, 300], [1500, 300]],
            [[1600, 300], [1700, 300]],
            [[100, 200], [200, 200]],
            [[500, 200], [600, 200]],
            [[800, 200], [1100, 200]],
            [[1200, 200], [1400, 200]],
            [[1500, 200], [1600, 200]],
            [[200, 100], [300, 100]],
            [[500, 100], [800, 100]],
            [[1000, 100], [1200, 100]],
            [[1400, 100], [1600, 100]],
            [[1800, 100], [1900, 100]],
            [[0, 0], [2000, 0]],
            # vertical
            [[0, 0], [0, 1300]],
            [[100, 0], [100, 300]],
            [[100, 400], [100, 1000]],
            [[200, 300], [200, 400]],
            [[200, 600], [200, 800]],
            [[200, 900], [200, 1000]],
            [[300, 100], [300, 600]],
            [[300, 800], [300, 1100]],
            [[400, 0], [400, 300]],
            [[400, 1200], [400, 1300]],
            [[500, 100], [500, 200]],
            [[600, 200], [600, 400]],
            [[600, 1100], [600, 1200]],
            [[700, 200], [700, 300]],
            [[700, 400], [700, 1100]],
            [[800, 100], [800, 200]],
            [[800, 300], [800, 500]],
            [[800, 600], [800, 700]],
            [[800, 1000], [800, 1100]],
            [[900, 0], [900, 100]],
            [[900, 300], [900, 600]],
            [[900, 900], [900, 1200]],
            [[1000, 100], [1000, 200]],
            [[1200, 100], [1200, 200]],
            [[1300, 0], [1300, 100]],
            [[1400, 100], [1400, 700]],
            [[1500, 700], [1500, 1000]],
            [[1500, 1100], [1500, 1200]],
            [[1600, 200], [1600, 400]],
            [[1600, 600], [1600, 700]],
            [[1600, 1000], [1600, 1100]],
            [[1600, 1200], [1600, 1300]],
            [[1700, 1100], [1700, 1200]],
            [[1700, 700], [1700, 800]],
            [[1700, 500], [1700, 600]],
            [[1700, 0], [1700, 300]],
            [[1800, 100], [1800, 400]],
            [[1800, 600], [1800, 700]],
            [[1800, 900], [1800, 1200]],
            [[1900, 800], [1900, 1300]],
            [[1900, 100], [1900, 600]],
            [[2000, 0], [2000, 1300]],
        ])
        rooms = [
                # [[300, 500], 400, 600],
                #  [[900, 800], 600, 400],
                #  [[900, 300], 500, 400],
                 ]
        ax.set_xlim([-margin, 2000 + margin])
        ax.set_ylim([-margin, 1300 + margin])

    if means is not None:
        walls -= means['pose'][:, :, :2]
    if stds is not None:
        walls /= stds['pose'][:, :, :2]
    # color = (0.8, 0.8, 0.8)
    color = (0, 0, 0)

    ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color=color, linewidth=linewidth)
    for room in rooms:
        ax.add_patch(Rectangle(*room, facecolor=(0.85, 0.85, 0.85), linewidth=0))
    ax.set_aspect('equal')


def plot_trajectories(data, figure_name=None, show=False, pause=False, emphasize=None, odom=False, mincolor=0.0, linewidth=0.3):
    from methods.odom import OdometryBaseline
    if figure_name is not None:
        plt.figure(figure_name)
    for i, trajectories in enumerate(data['s']):
        color = np.random.uniform(low=mincolor, high=1.0, size=3)
        plt.plot(trajectories[:, 0], trajectories[:, 1], color=color, linewidth=linewidth, zorder=0)
    if emphasize is not None:
        true_traj = data['s'][emphasize, :20, :]
        odom = OdometryBaseline()
        odom_traj = odom.predict(None, {k:data[k][emphasize:emphasize+1, :20] for k in data.keys()})[0]
        print(true_traj)
        print(odom_traj)

        traj = odom_traj
        plt.plot(traj[:, 0], traj[:, 1], '--', color=[0.0, 0.0, 1.0], linewidth=0.8, zorder=0)
        # plt.plot(traj[:, 0], traj[:, 1], 'o', markerfacecolor='None',
        #     markeredgecolor=[0.0, 0.0, 0.0],
        #     markersize=5)
        # plt.quiver(traj[:, 0], traj[:, 1], np.cos(traj[:, 2]), np.sin(traj[:, 2]),
        #            color=[0.0, 0.0, 0.0], zorder=1, headlength=0, headaxislength=0, scale=10, width=0.02, units='inches', scale_units='inches')

        traj = true_traj
        plt.plot(traj[:, 0], traj[:, 1], '-', color=[0.0, 0.0, 1.0], linewidth=0.8, zorder=0)
        plt.plot(traj[:, 0], traj[:, 1], 'o', markerfacecolor='None',
            markeredgecolor=[0.0, 0.0, 0.0],
            markersize=5)
        plt.quiver(traj[:, 0], traj[:, 1], np.cos(traj[:, 2]), np.sin(traj[:, 2]),
                   color=[0.0, 0.0, 0.0], zorder=1, headlength=0, headaxislength=0, scale=10, width=0.02, units='inches', scale_units='inches')

    plt.gca().set_aspect('equal')
    show_pause(show, pause)


def plot_trajectory(data, figure_name=None, show=False, pause=False, emphasize=None, odom=False, mincolor=0.0, linewidth=0.3):
    from methods.odom import OdometryBaseline
    if figure_name is not None:
        plt.figure(figure_name)
    for i, trajectories in enumerate(data['s']):
        plt.plot(trajectories[:, 0], trajectories[:, 1], '-', color='red', linewidth=linewidth, zorder=0, markersize=4)
        plt.plot(trajectories[:5, 0], trajectories[:5, 1], '.', color='blue', linewidth=linewidth, zorder=0, markersize=8)
        plt.plot(trajectories[0, 0], trajectories[0, 1], '.', color='blue', linewidth=linewidth, zorder=0, markersize=16)

        # plt.quiver(trajectories[:5, 0], trajectories[:5, 1],
        #        np.cos(trajectories[:5, 2]), np.sin(trajectories[:5, 2]),
        #            # np.arange(len(trajectories)), cmap='viridis', alpha=1.0,
        #            color='red', alpha=1.0,
        #        **quiv_kwargs
        #        )

    plt.gca().set_aspect('equal')
    show_pause(show, pause)


def plot_observations(data, n=20, figure_name=None, show=False, pause=False):

    plt.figure(figsize=(10,2.5))
    for i in range(n):
        # plt.figure('Normalized image')
        # plt.gca().clear()
        # plt.imshow(0.5 + rgbds[i, :, :, :3]/10, interpolation='nearest')
        # plt.pause(0.001)
        #
        # plt.figure('Depth image')
        # plt.gca().clear()
        # plt.imshow(0.5 + rgbds[i, :, :, 3] / 10, interpolation='nearest', cmap='coolwarm', vmin=0.0, vmax=1.0)
        # plt.pause(0.001)


        # plt.gca().clear()
        # plt.subplot(2, 10, i+1)
        plt.subplot(1, n, i+1)
        plt.imshow(np.clip(data['o'][0, i, :, :, :]/255.0, 0.0, 1.0), interpolation='nearest')
        plt.axis('off')
        # plt.tight_layout(pad=0.1)
        # plt.pause(0.1)
    show_pause(show, pause)


def view_data(data):
    # overview plot
    for poses in data['s']:
        plt.figure('Overview')
        plt.plot(poses[:, 0], poses[:, 1])

        # # sample plot
        # for poses, velocities, rgbds in zip(data['pose'], data['vel'], data['rgbd']):
        #     # for poses in data['pose']:
        #     plt.ioff()
        #     plt.figure('Sample')
        #     # plt.plot(poses[:, 0], 'r-')
        #     # plt.plot(poses[:, 1], 'g-')
        #     plt.plot(poses[:, 2], 'b-')
        #     # plt.plot(velocities[:, 0], 'r--')
        #     # plt.plot(velocities[:, 1], 'g--')
        #     plt.plot(velocities[:, 2], 'b--')
        #     plt.show()
        #
        #     # for i in range(100):
        #     #     plt.figure('Normalized image')
        #     #     plt.gca().clear()
        #     #     plt.imshow(0.5 + rgbds[i, :, :, :3]/10, interpolation='nearest')
        #     #     plt.pause(0.001)
        #     #
        #     #     plt.figure('Depth image')
        #     #     plt.gca().clear()
        #     #     plt.imshow(0.5 + rgbds[i, :, :, 3] / 10, interpolation='nearest', cmap='coolwarm', vmin=0.0, vmax=1.0)
        #     #     plt.pause(0.001)
        #     #
        #     #     plt.figure('Real image')
        #     #     plt.gca().clear()
        #     #     plt.imshow((rgbds*stds['rgbd'][0] + means['rgbd'][0])[i, :, :, :3]/255.0, interpolation='nearest')
        #     #     plt.pause(0.1)
