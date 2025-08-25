from d2c import d2c
import numpy as np

A = np.array([[0.8, 0.1], [0.0, 0.9]])
B = np.array([[0.1], [0.05]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

dt = 0.01

A_c, B_d, C_d, D_d = d2c(A, B, C, D, dt)

print(f"A_c: {A_c}, B_d: {B_d}, C_d: {C_d}, D_d: {D_d}")