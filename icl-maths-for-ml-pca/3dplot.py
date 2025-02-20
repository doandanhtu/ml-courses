import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the vectors and subspaces
v = np.array([12, 0, 0])
u1 = np.array([1, 1, 1])
u2 = np.array([0, 1, 2])

# Create matrix U with u1 and u2 as columns (for subspace U)
U = np.column_stack((u1, u2))

# Compute the projection of v onto U
# v_proj = U * ((Uᵀ U)⁻¹ * Uᵀ * v)
v_proj = U @ np.linalg.inv(U.T @ U) @ (U.T @ v)
print("Projection of v onto U:", v_proj)

# Define the additional subspace vector (line)
u3 = -np.sqrt(6) * np.array([10, 4, -2])
print("Vector spanning the additional subspace (u3):", u3)

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original vector v (red arrow)
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r', arrow_length_ratio=0.1, label='v = [12, 0, 0]')

# Plot the projected vector v_proj (green arrow)
ax.quiver(0, 0, 0, v_proj[0], v_proj[1], v_proj[2], color='g', arrow_length_ratio=0.1, label='Projection of v onto U')

# Plot the plane representing subspace U
# Generate a grid using parameters a and b for u1 and u2
a_vals = np.linspace(-5, 5, 10)
b_vals = np.linspace(-5, 5, 10)
A, B = np.meshgrid(a_vals, b_vals)
X = A * u1[0] + B * u2[0]
Y = A * u1[1] + B * u2[1]
Z = A * u1[2] + B * u2[2]
ax.plot_surface(X, Y, Z, color='b', alpha=0.2, edgecolor='none', label='Subspace U')

# Plot the additional subspace (a line) spanned by u3
# Generate points along the line: t in [-1, 1] (adjust t-range if needed)
t = np.linspace(-1, 1, 10)
u3_line = np.outer(t, u3)  # Each row is a point on the line
ax.plot(u3_line[:, 0], u3_line[:, 1], u3_line[:, 2],
        color='m', linewidth=3,
        label=r'Subspace spanned by $-\sqrt{6}[10,4,-2]$')

# Adjust axes limits to cover all vectors and subspace elements
# Gather key points: v, v_proj, and the endpoints of the u3 line
points = np.vstack((v, v_proj, u3_line[0], u3_line[-1]))
max_val = np.max(np.abs(points)) + 5  # add a small margin
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

# Label axes and add title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Projection of v onto U and Additional Subspace')

# Add legend (Note: legend for surfaces may not appear automatically, so we create custom handles)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='r', lw=3),
                Line2D([0], [0], color='g', lw=3),
                Line2D([0], [0], color='b', lw=6, alpha=0.2),
                Line2D([0], [0], color='m', lw=3)]
ax.legend(custom_lines, ['v = [12, 0, 0]',
                         'Projection of v onto U',
                         'Subspace U (plane)',
                         r'Subspace spanned by $-\sqrt{6}[10,4,-2]$'])

plt.show()
