import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# ------ A -------
delta = 0.025
x = np.arange(-2, 4, delta)
y = np.arange(-2, 4, delta)

X, Y = np.meshgrid(x, y)

Z = mlab.bivariate_normal(X, Y, 1, np.sqrt(2), 1, 1, 0)

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Part a')
plt.show()

# ------ B -------
delta = 0.025
x = np.arange(-4, 2, delta)
y = np.arange(-2, 6, delta)

X, Y = np.meshgrid(x, y)

Z = mlab.bivariate_normal(X, Y, np.sqrt(2), np.sqrt(3), -1, 2, 1)

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Part b')
plt.show()

# ------ C -------
delta = 0.025
x = np.arange(-4, 5, delta)
y = np.arange(-2, 4, delta)

X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, np.sqrt(2), 1, 0, 2, 1)
Z2 = mlab.bivariate_normal(X, Y, np.sqrt(2), 1, 2, 0, 1)

Z = Z1 - Z2

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Part c')
plt.show()

# ------ D -------
delta = 0.025
x = np.arange(-5, 5, delta)
y = np.arange(-5, 5, delta)

X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, np.sqrt(2), 1, 0, 2, 1)
Z2 = mlab.bivariate_normal(X, Y, np.sqrt(2), np.sqrt(3), 2, 0, 1)

Z = Z1 - Z2

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Part d')
plt.show()

# ------ E -------
delta = 0.025
x = np.arange(-5, 5, delta)
y = np.arange(-5, 5, delta)

X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, np.sqrt(2), 1, 1, 1, 0)
Z2 = mlab.bivariate_normal(X, Y, np.sqrt(2), np.sqrt(2), -1, -1, 1)

Z = Z1 - Z2

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Part e')
plt.show()
