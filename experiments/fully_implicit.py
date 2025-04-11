import numpy as np

x_len = 4
y_len = 4
z_len = 4

diffusion_coefficient = 10000
decay_rate = 0
dspace = 20
dtime = 0.01

A = np.zeros((x_len * y_len * z_len, x_len * y_len * z_len))

# create fully implicit diffusion+decay matrix
for z in range(z_len):
    for y in range(y_len):
        for x in range(x_len):
            i = z * y_len * x_len + y * x_len + x
            
            r = dtime * diffusion_coefficient / dspace**2

            A[i, i] = 1 + dtime * decay_rate

            neighbors_count = 0

            if x > 0:
                A[i, i - 1] = -r
                neighbors_count += 1
            if x < x_len - 1:
                A[i, i + 1] = -r
                neighbors_count += 1
            if y > 0:
                A[i, i - x_len] = -r
                neighbors_count += 1
            if y < y_len - 1:
                A[i, i + x_len] = -r
                neighbors_count += 1
            if z > 0:
                A[i, i - x_len * y_len] = -r
                neighbors_count += 1
            if z < z_len - 1:
                A[i, i + x_len * y_len] = -r
                neighbors_count += 1

            A[i, i] += r * neighbors_count

# for row in matrix:
#     print(' '.join(f'{val:.2f}' for val in row))

# print('\n')

# cholesky_decomposition = np.linalg.cholesky(A)

# for row in cholesky_decomposition:
#     print(' '.join(f'{val:.2f}' for val in row))

# print('\n')

b = np.ones((x_len * y_len * z_len))

for z in range(z_len):
    for y in range(y_len):
        for x in range(x_len):
            i = z * y_len * x_len + y * x_len + x
            b[i] = i

x = np.linalg.solve(A, b)

print(x)
