import numpy as np

x_len = 4
y_len = 4
z_len = 4

diffusion_coefficient = 10000
decay_rate = 0
dspace = 20
dtime = 0.01
iterations = 1

b = np.ones((x_len * y_len * z_len))

for z in range(z_len):
    for y in range(y_len):
        for x in range(x_len):
            i = z * y_len * x_len + y * x_len + x
            b[i] = i

b2 = np.ones((x_len * y_len * z_len))

for _ in range(iterations):
    for z in range(z_len):
        for y in range(y_len):
            for x in range(x_len):
                i = z * y_len * x_len + y * x_len + x
                r = dtime * diffusion_coefficient / dspace**2
                tmp = 0
                neighbors_count = 0
                if x > 0:
                    tmp += b[i - 1]
                    neighbors_count += 1
                if x < x_len - 1:
                    tmp += b[i + 1]
                    neighbors_count += 1
                if y > 0:
                    tmp += b[i - x_len]
                    neighbors_count += 1
                if y < y_len - 1:
                    tmp += b[i + x_len]
                    neighbors_count += 1
                if z > 0:
                    tmp += b[i - x_len * y_len]
                    neighbors_count += 1
                if z < z_len - 1:
                    tmp += b[i + x_len * y_len]
                    neighbors_count += 1
                b2[i] = b[i] + r * (tmp - neighbors_count * b[i]) - dtime * decay_rate * b[i]
    b, b2 = b2, b

print(b)
