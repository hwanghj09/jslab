import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

def macroscopic(fin, nx, ny, ci):
    rho = np.sum(fin, axis=0)  # rho(밀도) 구하는 곳
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += ci[i, 0] * fin[i, :, :]
        u[1, :, :] += ci[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u

def equilibrium(rho, u, ci, wi, nx, ny):
    usqr = (3 / 2) * (u[0]**2 + u[1]**2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        cu = 3 * (ci[i, 0] * u[0, :, :] + ci[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * wi[i] * (1 + cu + 0.5 * cu**2 - usqr)
    return feq

Re = 200
maxInter = 500000
nx, ny = 105, 45
ly = ny - 1
uLB = 0.01
cx, cy, r = nx // 6, ny // 2, ny // 11
nulb = 1 * uLB * r / Re
omega = 1 / (3 * nulb + 0.5)

ci = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]])
wi = np.array([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 4 / 9])

col_left = np.array([0, 4, 7])
col_rest = np.array([1, 3, 8])
col_right = np.array([2, 5, 6])
col_wall = np.array([2, 3, 0, 1, 5, 6, 7, 6, 8])

obstacle = np.empty((nx, ny), dtype=bool)
for i in range(nx):
    for j in range(ny):
        if (i - cx)**2 + (j - cy)**2 <= r**2:
            obstacle[i, j] = True
        else:
            obstacle[i, j] = False

u = np.zeros((2, nx, ny))
u[0, :, :] = uLB

fin = equilibrium(1, u, ci, wi, nx, ny)

# Enable interactive mode for real-time plotting
plt.ion()

for time in tqdm(range(maxInter)):
    fin[col_right, -1, :] = fin[col_right, -2, :]

    rho, u = macroscopic(fin, nx, ny, ci)

    u[0, 0, :] = uLB
    u[1, 0, :] = 0.0
    rho[0, :] = 1 / (1 - u[0, 0, :]) * (np.sum(fin[col_rest, 0, :], axis=0) + 2 * np.sum(fin[col_right, 0, :], axis=0))

    feq = equilibrium(rho, u, ci, wi, nx, ny)
    fin[col_left, 0, :] = feq[col_left, 0, :] + fin[col_right, 0, :] - feq[col_right, 0, :]

    fout = fin - omega * (fin - feq)

    for i in range(9):
        fout[i, obstacle] = fin[col_wall[i], obstacle]
    for i in range(9):
        fin[i, :, :] = np.roll(np.roll(fout[i, :, :], ci[i, 0], axis=0), ci[i, 1], axis=1)

    # Plot every 100 iterations
    if time % 100 == 0:
        plt.clf()  # Clear the previous figure
        plt.matshow(np.transpose(np.sqrt(u[0]**2 + u[1]**2)), cmap=cm.Reds, fignum=False)
        plt.colorbar()
        plt.pause(0.0001)  # Short pause to allow for the figure to update

plt.ioff()  # Disable interactive mode after the loop
plt.show()  # Show the final plot
