import numpy as np
import bhtsne

np.random.seed(1137)
X = np.random.rand(500, 10)
proj, betas = bhtsne.run_bh_tsne(X, verbose=True, randseed=1137, return_betas=True)

assert betas.shape[0] == 500
assert proj[0,0] == -7.235696435669544
