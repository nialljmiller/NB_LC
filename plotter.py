import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ----- Visualization Class -----
class Visualizer:
    def __init__(self, trajectories, lightcurve_times, lightcurve_values):
        self.trajectories = trajectories
        self.lightcurve_times = lightcurve_times
        self.lightcurve_values = lightcurve_values

    def plot_orbits(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, traj in self.trajectories.items():
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Body {i}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Orbits of Bodies')
        plt.show()

    def plot_lightcurve(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.lightcurve_times, self.lightcurve_values, 'k-')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Simulated Light Curve')
        plt.show()


    def generate_animation(self,trajectories, lightcurve_times, lightcurve_values, filename='animation.gif'):
        """
        Generates a GIF animation with a left panel showing 3D orbits and a right panel showing the light curve.
        The GIF iterates over the simulation time steps.
        """
        num_frames = len(lightcurve_times)
        fig = plt.figure(figsize=(12, 6))
        
        # Left panel: 3D orbits
        ax1 = fig.add_subplot(121, projection='3d')
        # Right panel: light curve
        ax2 = fig.add_subplot(122)
        
        # Prepare line objects for each body's orbit and current position marker
        body_lines = {}
        body_markers = {}
        for key in trajectories.keys():
            line, = ax1.plot([], [], [], label=f'Body {key}')
            marker, = ax1.plot([], [], [], 'o')
            body_lines[key] = line
            body_markers[key] = marker

        # Prepare light curve line and current time marker
        lc_line, = ax2.plot([], [], 'k-', lw=2)
        current_time_line = ax2.axvline(x=lightcurve_times[0], color='r', linestyle='--', lw=1)

        # Set axis labels and titles
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Orbits of Bodies')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Flux')
        ax2.set_title('Simulated Light Curve')

        # Compute global limits for the 3D plot
        all_positions = np.concatenate([np.array(traj) for traj in trajectories.values()])
        x_min, y_min, z_min = all_positions.min(axis=0)
        x_max, y_max, z_max = all_positions.max(axis=0)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax1.legend()
        
        def update(frame):
            # Update 3D orbits and markers
            for key in trajectories.keys():
                traj = np.array(trajectories[key])
                if frame < len(traj):
                    traj_frame = traj[:frame+1]
                    body_lines[key].set_data(traj_frame[:, 0], traj_frame[:, 1])
                    body_lines[key].set_3d_properties(traj_frame[:, 2])
                    # Wrap the scalar coordinates in lists to satisfy set_data()
                    body_markers[key].set_data([traj_frame[-1, 0]], [traj_frame[-1, 1]])
                    body_markers[key].set_3d_properties([traj_frame[-1, 2]])
            # Update light curve up to current frame
            t_data = lightcurve_times[:frame+1]
            flux_data = lightcurve_values[:frame+1]
            lc_line.set_data(t_data, flux_data)
            ax2.relim()
            ax2.autoscale_view()
            current_time_line.set_xdata([lightcurve_times[frame], lightcurve_times[frame]])
            return list(body_lines.values()) + list(body_markers.values()) + [lc_line, current_time_line]
        
        ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
        ani.save(filename, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Animation saved to {filename}")
