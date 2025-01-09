import numpy as np

M = np.array([[0.1, 0.7, 0.1, 0.1],
              [0.7, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.7],
              [0.1, 0.1, 0.7, 0.1]])
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)

det = np.linalg.det(M)
print(det)