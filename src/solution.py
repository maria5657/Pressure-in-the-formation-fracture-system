from numpy.typing import NDArray
import numpy as np
from src.constants import PhysicalConstants
from src.inv_laplace import get_inv_laplace
from src.solution_in_laplace import SolutionInLaplaceDomain
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, constants: PhysicalConstants = PhysicalConstants()):
        self.constants = constants
        self.laplace_domain_solver = SolutionInLaplaceDomain(self.constants)

    def solve(self, x: float, t_moments: NDArray):
        x_tilda = self.constants.phi * x + self.constants.psi
        laplace_func = self.laplace_domain_solver.get_analytic_solution_func(x_tilda=x_tilda)
        p = get_inv_laplace(t_moments=t_moments, func=laplace_func)
        return p

def hours2seconds(hours: NDArray | list):
    return np.array(list(map(lambda t: t * 3600, hours)))

def p_line2p(p: NDArray | list, constants: PhysicalConstants = PhysicalConstants()):
    return np.array(list(map(lambda t: t + constants.p_init, p)))


if __name__ == '__main__':
    solver = Solver()
    t_moments = np.exp(np.linspace(np.log(0.1), np.log(10), num=100))
    p = solver.solve(x=1e-10, t_moments=hours2seconds(t_moments))
    coef = np.polyfit(t_moments, p, 4)
    p_new = np.poly1d(coef)
    plt.plot(t_moments, p, label='решение')

    t = np.linspace(0.1, 10, 1000)
    plt.plot(t, p_new(t), label='апроксимация')

    plt.xlabel('Время, часы')
    plt.ylabel(f'Давление от {PhysicalConstants.p_init}, Па')
    plt.legend()
    plt.grid()
    plt.show()
