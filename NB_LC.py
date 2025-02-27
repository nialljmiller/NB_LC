import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----- Star Class -----
class Star:
    def __init__(self, mass, radius, luminosity, limb_darkening=0.6, reflection_coeff=0.1):
        """
        mass: in solar masses (arbitrary units)
        radius: in solar radii (arbitrary units)
        luminosity: normalized luminosity
        limb_darkening: linear limb darkening coefficient (0 = none, 1 = fuck-all, 1 = strong)
        reflection_coeff: crude reflection coefficient
        """
        self.mass = mass
        self.radius = radius
        self.luminosity = luminosity
        self.limb_darkening = limb_darkening
        self.reflection_coeff = reflection_coeff

# ----- Body Class (for dynamics) -----
class Body(Star):
    def __init__(self, mass, radius, luminosity, limb_darkening=0.6, reflection_coeff=0.1, position=None, velocity=None):
        super().__init__(mass, radius, luminosity, limb_darkening, reflection_coeff)
        self.position = np.array(position, dtype=float) if position is not None else np.zeros(3)
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)

# ----- N-Body Simulator -----
class NBodySimulator:
    def __init__(self, bodies, G=1.0):
        """
        bodies: list of Body instances
        G: gravitational constant (arbitrary units)
        """
        self.bodies = bodies
        self.G = G

    def compute_accelerations(self):
        n = len(self.bodies)
        accelerations = [np.zeros(3) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r_vec = self.bodies[j].position - self.bodies[i].position
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-6:
                    accelerations[i] += self.G * self.bodies[j].mass * r_vec / r_mag**3
        return accelerations

    def step(self, dt):
        # Leapfrog integration for decent energy conservation over long runs
        acc = self.compute_accelerations()
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * dt * acc[i]
        for body in self.bodies:
            body.position += dt * body.velocity
        acc_new = self.compute_accelerations()
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * dt * acc_new[i]

    def simulate(self, t_total, dt):
        num_steps = int(t_total / dt)
        trajectories = {i: [] for i in range(len(self.bodies))}
        for _ in range(num_steps):
            for i, body in enumerate(self.bodies):
                trajectories[i].append(body.position.copy())
            self.step(dt)
        return trajectories

# ----- Light Curve Simulator for N Bodies -----
class LightCurveSimulatorN:
    def __init__(self, bodies, num_grid=50):
        """
        bodies: list of Body instances (with positions updated by NBodySimulator)
        num_grid: grid resolution for each star's disk integration
        """
        self.bodies = bodies
        self.num_grid = num_grid

    def compute_flux(self):
        """
        Computes the total flux as seen by an observer along the +z axis.
        For each star, we compute its visible fraction by checking occlusion by any star in front.
        """
        # Sort bodies by their z coordinate (largest z = closest to the observer)
        sorted_bodies = sorted(self.bodies, key=lambda b: b.position[2], reverse=True)
        total_flux = 0.0
        # For each star, occluders are those in front (with higher z)
        for i, body in enumerate(sorted_bodies):
            occluders = sorted_bodies[:i]
            visible_fraction = self._compute_visible_fraction(body, occluders)
            # Basic flux from the star (luminosity scaled by visible fraction)
            flux = body.luminosity * visible_fraction

            # Optional: crude reflection effects from occluders (if you wanna crank it up)
            for occ in occluders:
                separation = np.linalg.norm((np.array([body.position[0], body.position[1]]) -
                                              np.array([occ.position[0], occ.position[1]])))
                flux += body.reflection_coeff * occ.luminosity / (separation**2 + 1e-6)
            total_flux += flux
        return total_flux

    def _compute_visible_fraction(self, body, occluders):
        """
        For a given body, integrate over its disk (projected in the (x,y) plane) to find the fraction not occluded.
        Applies a linear limb darkening law: I = 1 - u*(1 - sqrt(1 - r^2))
        """
        num_points = self.num_grid
        theta = np.linspace(0, 2*np.pi, num_points)
        r = np.linspace(0, body.radius, num_points)
        R, Theta = np.meshgrid(r, theta)
        # Disk coordinates relative to star center
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        # Normalized radial coordinate for limb darkening
        r_norm = R / body.radius
        intensity = 1 - body.limb_darkening * (1 - np.sqrt(1 - r_norm**2))
        total_intensity = np.sum(intensity)
        # Assume the star's disk is centered at (body.position[0], body.position[1])
        visible_mask = np.ones_like(intensity, dtype=bool)
        for occ in occluders:
            # Compute offset of occluder center from current star center in the (x,y) plane
            dx = (X + body.position[0]) - occ.position[0]
            dy = (Y + body.position[1]) - occ.position[1]
            dist = np.sqrt(dx**2 + dy**2)
            visible_mask &= (dist > occ.radius)
        visible_intensity = np.sum(intensity * visible_mask)
        return visible_intensity / total_intensity if total_intensity > 0 else 0

# ----- Visualization Class -----
class Visualizer:
    def __init__(self, trajectories, lightcurve_times, lightcurve_values):
        """
        trajectories: dict with keys as body indices and values as lists of 3D positions over time
        lightcurve_times: list/array of times
        lightcurve_values: list/array of total flux values at each time
        """
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

# ----- Main Application -----
def main():
    # Define a few bodies with some initial positions and velocities.
    # Feel free to tweak these parameters to see different crazy-ass dynamics.
    bodies = []
    bodies.append(Body(mass=1.0, radius=1.0, luminosity=1.0, position=[-5, 0, 5], velocity=[0, 0.5, 0]))
    bodies.append(Body(mass=0.8, radius=0.8, luminosity=0.5, position=[5, 0, 3], velocity=[0, -0.5, 0]))
    #bodies.append(Body(mass=0.5, radius=0.6, luminosity=0.3, position=[0, 5, 1], velocity=[-0.5, 0, 0]))
    
    # Set up the N-body simulator with a chosen gravitational constant
    nbody = NBodySimulator(bodies, G=1.0)
    
    t_total = 200.0
    dt = 0.1
    num_steps = int(t_total / dt)
    
    trajectories = {i: [] for i in range(len(bodies))}
    lightcurve_times = []
    lightcurve_values = []
    lc_sim = LightCurveSimulatorN(bodies, num_grid=50)
    
    t = 0
    for step in range(num_steps):
        # Record positions for each body
        for i, body in enumerate(bodies):
            trajectories[i].append(body.position.copy())
        # Compute current flux (light curve) from the system
        flux = lc_sim.compute_flux()
        lightcurve_times.append(t)
        lightcurve_values.append(flux)
        nbody.step(dt)
        t += dt

    # Visualize the trajectories and the light curve
    viz = Visualizer(trajectories, lightcurve_times, lightcurve_values)
    viz.plot_orbits()
    viz.plot_lightcurve()

if __name__ == '__main__':
    main()
