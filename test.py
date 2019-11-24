import numpy as np
import bhtsne

np.random.seed(1137)
X = np.random.rand(500, 10)
proj, betas, cpp, cpi = bhtsne.run_bh_tsne(X, verbose=False, randseed=1137,
    return_betas=True, return_cost_per_point=True, return_cost_per_iter=True)

assert betas.shape[0] == 500
assert cpp.shape[0] == 500
assert cpi.shape[0] == 1000
assert proj[0,0] == -7.235696435669544
