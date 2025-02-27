import numpy as np
import plotter as plotter
import stellar_star as star
import lightcurve_sim as lc_sim

# ----- N-Body Simulator with Advanced Options -----
class NBodySimulator:
    def __init__(self, bodies, G=1.0, softening=1e-3, integrator='leapfrog'):
        """
        bodies: list of Body instances
        G: gravitational constant
        softening: softening parameter to avoid singularities
        integrator: 'leapfrog' or 'rk4'
        """
        self.bodies = bodies
        self.G = G
        self.softening = softening
        self.integrator = integrator

    def compute_accelerations(self):
        n = len(self.bodies)
        accelerations = []
        for i in range(n):
            a = np.zeros(3)
            for j in range(n):
                if i == j:
                    continue
                r_vec = self.bodies[j].position - self.bodies[i].position
                r2 = np.dot(r_vec, r_vec) + self.softening**2
                r3 = r2 * np.sqrt(r2)
                a += self.G * self.bodies[j].mass * r_vec / r3
            accelerations.append(a)
        return accelerations

    def step(self, dt):
        if self.integrator == 'leapfrog':
            self.step_leapfrog(dt)
        elif self.integrator == 'rk4':
            self.step_rk4(dt)
        else:
            raise ValueError("Unknown integrator. Use 'leapfrog' or 'rk4'.")

    def step_leapfrog(self, dt):
        # Half-step velocity update
        a = self.compute_accelerations()
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * dt * a[i]
        # Full-step position update
        for body in self.bodies:
            body.position += dt * body.velocity
        # Another half-step velocity update
        a_new = self.compute_accelerations()
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * dt * a_new[i]

    def step_rk4(self, dt):
        n = len(self.bodies)
        # Save initial state
        positions0 = [body.position.copy() for body in self.bodies]
        velocities0 = [body.velocity.copy() for body in self.bodies]

        def compute_accel(positions):
            acc = []
            for i in range(n):
                a = np.zeros(3)
                for j in range(n):
                    if i == j:
                        continue
                    r_vec = positions[j] - positions[i]
                    r2 = np.dot(r_vec, r_vec) + self.softening**2
                    r3 = r2 * np.sqrt(r2)
                    a += self.G * self.bodies[j].mass * r_vec / r3
                acc.append(a)
            return acc

        # k1
        a1 = compute_accel(positions0)
        k1_x = [v.copy() for v in velocities0]
        k1_v = [a.copy() for a in a1]

        # k2
        positions_k2 = [positions0[i] + 0.5 * dt * k1_x[i] for i in range(n)]
        velocities_k2 = [velocities0[i] + 0.5 * dt * k1_v[i] for i in range(n)]
        a2 = compute_accel(positions_k2)
        k2_x = [v.copy() for v in velocities_k2]
        k2_v = [a.copy() for a in a2]

        # k3
        positions_k3 = [positions0[i] + 0.5 * dt * k2_x[i] for i in range(n)]
        velocities_k3 = [velocities0[i] + 0.5 * dt * k2_v[i] for i in range(n)]
        a3 = compute_accel(positions_k3)
        k3_x = [v.copy() for v in velocities_k3]
        k3_v = [a.copy() for a in a3]

        # k4
        positions_k4 = [positions0[i] + dt * k3_x[i] for i in range(n)]
        velocities_k4 = [velocities0[i] + dt * k3_v[i] for i in range(n)]
        a4 = compute_accel(positions_k4)
        k4_x = [v.copy() for v in velocities_k4]
        k4_v = [a.copy() for a in a4]

        # Combine to update state
        for i in range(n):
            self.bodies[i].position = positions0[i] + (dt/6.0) * (k1_x[i] + 2*k2_x[i] + 2*k3_x[i] + k4_x[i])
            self.bodies[i].velocity = velocities0[i] + (dt/6.0) * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])

    def simulate(self, t_total, dt):
        num_steps = int(t_total / dt)
        trajectories = {i: [] for i in range(len(self.bodies))}
        for _ in range(num_steps):
            for i, body in enumerate(self.bodies):
                trajectories[i].append(body.position.copy())
            self.step(dt)
        return trajectories


# ----- Main Application -----
def main():
    # Define bodies with initial conditions
    bodies = []
    bodies.append(star.Body(mass=1.0, radius=1.0, luminosity=1.0,
                       position=[-5, 0, 5], velocity=[0, 0.25, 0]))
    bodies.append(star.Body(mass=0.8, radius=0.8, luminosity=0.5,
                       position=[5, 0, 3], velocity=[0, -0.25, 0]))
    # Add more bodies as desired

    # Choose integrator: 'leapfrog' or 'rk4'
    nbody = NBodySimulator(bodies, G=1.0, softening=1e-3, integrator='rk4')

    t_total = 2000.0
    dt = 1
    num_steps = int(t_total / dt)

    trajectories = {i: [] for i in range(len(bodies))}
    lightcurve_times = []
    lightcurve_values = []
    lcsim = lc_sim.LightCurveSimulatorN(bodies, num_grid=50, observer_direction=np.array([0, 0, 1]))

    t = 0
    for step in range(num_steps):
        for i, body in enumerate(bodies):
            trajectories[i].append(body.position.copy())
        flux = lcsim.compute_flux()
        lightcurve_times.append(t)
        lightcurve_values.append(flux)
        nbody.step(dt)
        t += dt

    # Visualize static plots (optional)
    viz = plotter.Visualizer(trajectories, lightcurve_times, lightcurve_values)
    viz.plot_orbits()
    viz.plot_lightcurve()
    viz.generate_animation(trajectories, lightcurve_times, lightcurve_values, filename='animation.gif')

if __name__ == '__main__':
    main()
