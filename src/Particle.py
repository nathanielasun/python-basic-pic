"""
Author: Nathaniel Sun
Date: 2026-06-16
Description: 
    Particle class for 3D PIC simulation
    Contains the position, velocity, charge, and mass of the particle.
    Also contains particle update routines.
"""

import numpy as np

class Particle:
    def __init__(self, kind, x, y, z, vx, vy, vz, q, m):
        self.kind = kind
        self.pos = np.array([x, y, z])
        self.vel = np.array([vx, vy, vz])
        self.q = q
        self.m = m

    def move(self, dt):
        self.pos += self.vel * dt

    def get_position(self):
        return self.pos
    
    def get_velocity(self):
        return self.vel
    
    def get_charge(self):
        return self.q

    def get_mass(self):
        return self.m

    def get_kind(self):
        return self.kind

    def set_position(self, pos):
        self.pos = pos

    def set_velocity(self, vel):
        self.vel = vel

    def set_charge(self, q):
        self.q = q

    def set_mass(self, m):
        self.m = m

    def set_kind(self, kind):
        self.kind = kind