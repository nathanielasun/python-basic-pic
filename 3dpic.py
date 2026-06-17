import numpy

class simulation:
    def __init__(self, dt, tmax, nx, ny, nz):
        self.dt = dt
        self.tmax = tmax
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.particles = []
        
    def add_particle(self, particle):
        self.particles.append(particle)
