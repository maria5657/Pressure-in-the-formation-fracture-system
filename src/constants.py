from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    k_r: float = 1e-3
    k_f: float = 0.5e2
    phi_f: float = 0.9
    phi_r: float = 0.3
    c_ft: float = 1
    c_rt: float = 1e-4
    mu: float = 1
    h: float = 1e1
    q_init: float = - 1e3
    x_f: float = 0.6e2
    p_init: float = 1e10
    psi: float = 5 + 1e-10
    phi: float = - 1 / 12

    @property
    def calc_C_r(self):
        return self.mu * self.phi_r * self.c_rt / self.k_r

    @property
    def calc_C_f(self):
        return self.mu * self.phi_f * self.c_ft / (self.k_f * self.phi * self.phi)

    @property
    def calc_theta(self):
        return 2 * self.k_r / (self.k_f * self.phi * self.phi)

    @property
    def calc_xi(self):
        return self.mu * self.q_init / (2 * self.k_f * self.psi * self.h)
