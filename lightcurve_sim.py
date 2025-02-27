import numpy as np

# ----- Light Curve Simulator with Advanced Projection and Limb Darkening -----
class LightCurveSimulatorN:
    def __init__(self, bodies, num_grid=50, observer_direction=np.array([0,0,1]), limb_darkening_law=None):
        """
        bodies: list of Body instances (with updated positions)
        num_grid: grid resolution for disk integration
        observer_direction: direction from which the system is observed
        limb_darkening_law: function(r_norm, limb_darkening_coeff) -> intensity.
                            Defaults to linear limb darkening.
        """
        self.bodies = bodies
        self.num_grid = num_grid
        self.observer_direction = observer_direction / np.linalg.norm(observer_direction)
        if limb_darkening_law is None:
            # Linear limb darkening: I = 1 - u*(1 - sqrt(1 - r^2))
            self.limb_darkening_law = lambda r_norm, u: 1 - u * (1 - np.sqrt(1 - r_norm**2))
        else:
            self.limb_darkening_law = limb_darkening_law

        # Setup projection basis for observer's plane
        if np.allclose(self.observer_direction, np.array([0,0,1])):
            self.basis1 = np.array([1, 0, 0])
            self.basis2 = np.array([0, 1, 0])
        else:
            arbitrary = np.array([0,0,1])
            if np.allclose(self.observer_direction, arbitrary):
                arbitrary = np.array([1,0,0])
            self.basis1 = np.cross(self.observer_direction, arbitrary)
            self.basis1 /= np.linalg.norm(self.basis1)
            self.basis2 = np.cross(self.observer_direction, self.basis1)
            self.basis2 /= np.linalg.norm(self.basis2)

    def project_to_plane(self, position):
        # Project 3D position to 2D observer's plane using basis vectors
        x = np.dot(position, self.basis1)
        y = np.dot(position, self.basis2)
        return np.array([x, y])

    def _compute_visible_fraction(self, body, occluders):
        """
        Integrate over the star's disk in the observer plane to compute unoccluded flux fraction.
        """
        center = self.project_to_plane(body.position)
        num_points = self.num_grid
        theta = np.linspace(0, 2*np.pi, num_points)
        r = np.linspace(0, body.radius, num_points)
        R, Theta = np.meshgrid(r, theta)
        # Coordinates on the disk
        X = center[0] + R * np.cos(Theta)
        Y = center[1] + R * np.sin(Theta)
        r_norm = R / body.radius
        intensity = self.limb_darkening_law(r_norm, body.limb_darkening)
        total_intensity = np.sum(intensity)
        visible_mask = np.ones_like(intensity, dtype=bool)
        for occ in occluders:
            occ_center = self.project_to_plane(occ.position)
            dx = X - occ_center[0]
            dy = Y - occ_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            visible_mask &= (dist > occ.radius)
        visible_intensity = np.sum(intensity * visible_mask)
        return visible_intensity / total_intensity if total_intensity > 0 else 0

    def compute_flux(self):
        """
        Compute total flux as seen by an observer along the given direction.
        Bodies are sorted by depth (dot with observer_direction).
        """
        sorted_bodies = sorted(self.bodies, key=lambda b: np.dot(b.position, self.observer_direction), reverse=True)
        total_flux = 0.0
        for i, body in enumerate(sorted_bodies):
            occluders = sorted_bodies[:i]
            visible_fraction = self._compute_visible_fraction(body, occluders)
            flux = body.luminosity * visible_fraction
            for occ in occluders:
                body_center = self.project_to_plane(body.position)
                occ_center = self.project_to_plane(occ.position)
                separation = np.linalg.norm(body_center - occ_center)
                flux += body.reflection_coeff * occ.luminosity / (separation**2 + 1e-6)
            total_flux += flux
        return total_flux
