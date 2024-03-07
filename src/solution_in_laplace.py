from cmath import sqrt
from scipy.special import hyperu, hyp1f1
from mpmath import hyperu
from numpy import exp
from src.math import solve_lin_eq
from src.constants import PhysicalConstants

class SolutionInLaplaceDomain:
    def __init__(self, constants: PhysicalConstants):
        self.constants = constants
        self.C_r = constants.calc_C_r
        self.C_f = constants.calc_C_f
        self.theta = constants.calc_theta
        self.xi = constants.calc_xi
        self.alpha = self.calc_alpha
        self.gamma = self.calc_gamma

    @property
    def calc_alpha(self):
        return (1 + (self.theta / 2) * sqrt(self.C_r / self.C_f)).real

    @property
    def calc_gamma(self):
        return float(2)

    def calc_a(self, s: complex):
        return self.theta * sqrt(s * self.C_r)

    def calc_b(self, s: complex):
        return sqrt(s * self.C_f)

    def get_coefs(self, s: complex):
        b = self.calc_b(s)
        psi = self.constants.psi
        x_t_left = psi
        x_t_right = self.constants.phi * self.constants.x_f + self.constants.psi
        a1 = exp(-b * x_t_left) * (
            hyp1f1(self.alpha, self.gamma, 2 * b * x_t_left) * (1 - b * x_t_left) +
            (2 * b * x_t_left * self.alpha / self.gamma) * hyp1f1(self.alpha + 1, self.gamma + 1, 2 * b * x_t_left)
        )
        a2 = complex(exp(-b * x_t_left) * (
            hyperu(self.alpha, self.gamma, 2 * b * x_t_left) * (1 - b * x_t_left) -
            2 * b * x_t_left * self.alpha * hyperu(self.alpha + 1, self.gamma + 1, 2 * b * x_t_left)
        ))
        a3 = exp(-b * x_t_right) * (
            hyp1f1(self.alpha, self.gamma, 2 * b * x_t_right) * (1 - b * x_t_right) +
            (2 * b * x_t_right * self.alpha / self.gamma) * hyp1f1(self.alpha + 1, self.gamma + 1, 2 * b * x_t_right)
        )
        a4 = complex(exp(-b * x_t_right) * (
            hyperu(self.alpha, self.gamma, 2 * b * x_t_right) * (1 - b * x_t_right) -
            2 * b * x_t_right * self.alpha * hyperu(self.alpha + 1, self.gamma + 1, 2 * b * x_t_right)
        ))
        C1, C2 = solve_lin_eq(
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            b1=self.xi / self.constants.phi / s,
            b2=0 / self.constants.phi,
        )
        return C1, C2

    def get_analytic_solution(self, x_tilda: float, s: complex):
        C1, C2 = self.get_coefs(s=s)
        b = self.calc_b(s)
        return complex(C1 * x_tilda * exp(-b * x_tilda) * hyp1f1(self.alpha, self.gamma, 2 * b * x_tilda) +
                C2 * x_tilda * exp(-b * x_tilda) * hyperu(self.alpha, self.gamma, 2 * b * x_tilda))

    def get_analytic_solution_func(self, x_tilda: float):
        return lambda s: self.get_analytic_solution(x_tilda=x_tilda, s=s)

    def get_numerical_solution(self, x_tilda: float, s: complex):
        from scipy.integrate import solve_bvp
        import numpy as np
        a = self.calc_a(s=s)
        b = self.calc_b(s=s)
        N = 100
        x_arr = np.linspace(self.constants.psi, self.constants.phi * self.constants.x_f + self.constants.psi + 1e-10, N)
        x_arr = np.flip(x_arr)
        y_guess = np.zeros((2, N), dtype=float)

        def f(x, y):
            return [y[1], y[0] * (a + b * b * x) / x]

        def b_r(ya, yb):
            return [ya[1] - 0 / self.constants.phi, yb[1] - self.xi / self.constants.phi / s]

        sol = solve_bvp(f, b_r, x_arr, y_guess)
        return sol.y[0][np.where(sol.x >= x_tilda)[0][0]]


if __name__ == '__main__':
    Problem = SolutionInLaplaceDomain(PhysicalConstants())
    import numpy as np
    numer, analyt = [], []
    x_arr = np.linspace(1e-10, 5, num=10)
    for x in x_arr:
        numer.append(Problem.get_numerical_solution(x_tilda=x, s=2))
        analyt.append(Problem.get_analytic_solution(x_tilda=x, s=2))
    import matplotlib.pyplot as plt
    plt.plot(x_arr, numer, label='numerical')
    plt.plot(x_arr, analyt, label='analytical')
    plt.legend()
    plt.show()