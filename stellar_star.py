import numpy as np

# ----- Star Class -----
class Star:
    def __init__(self, mass, radius, luminosity, limb_darkening=0.6, reflection_coeff=0.1):
        """
        mass: in solar masses (arbitrary units)
        radius: in solar radii (arbitrary units)
        luminosity: normalized luminosity
        limb_darkening: limb darkening coefficient (linear by default)
        reflection_coeff: reflection coefficient
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