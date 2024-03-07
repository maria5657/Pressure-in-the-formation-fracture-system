from tqdm import tqdm
from numpy.typing import NDArray
from mpmath import *


mp.dps = 50
mp.pretty = True

def get_inv_laplace(t_moments: NDArray, func: callable):
    p = []
    for t_moment in tqdm(t_moments):
        p.append(invertlaplace(func, t_moment, method='talbot'))
    return p