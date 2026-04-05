import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from Hough_Transform import hough_circle_known_radius, plot_hough_results
from Ransac import ransac_circle_known_radius, plot_ransac_results
from linear_least_square import algebraic_least_squares
from radial_least_squares import radial_least_squares

# 1. Hough Transform Result
print("Generating Hough Transform result...")
true_center = np.array([0, 0])
true_radius = 5.0
n_points_h = 25
angles_h = np.linspace(0, 2*np.pi, n_points_h, endpoint=False)
points_h = true_center + true_radius * np.column_stack([np.cos(angles_h), np.sin(angles_h)])
points_h += np.random.normal(0, 0.2, (n_points_h, 2))

center_h, acc_h, x_bins_h, y_bins_h, votes_h = hough_circle_known_radius(points_h, true_radius)
plot_hough_results(points_h, true_radius, center_h, acc_h, x_bins_h, y_bins_h, votes_h)
plt.savefig("hough_result.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. RANSAC Result
print("Generating RANSAC result...")
true_center_r = np.array([2.5, 2.5])
true_radius_r = 4.5
n_inliers_r = 50
angles_r = np.linspace(0, 2*np.pi, n_inliers_r, endpoint=False)
inliers_r = true_center_r + true_radius_r * np.column_stack([np.cos(angles_r), np.sin(angles_r)])
inliers_r += np.random.normal(0, 0.1, (n_inliers_r, 2))
n_outliers_r = 30
outliers_r = np.random.uniform(-4, 10, (n_outliers_r, 2))
points_r = np.vstack([inliers_r, outliers_r])

center_r, inliers_mask_r, _ = ransac_circle_known_radius(points_r, true_radius_r, threshold=0.4, iterations=1000)
plot_ransac_results(points_r, true_radius_r, center_r, inliers_mask_r, "RANSAC Robust Circle Fit")
plt.savefig("ransac_result.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Radial Least Squares Result
print("Generating Radial LS result...")
# Use moderate noise for comparison
n_points_l = 30
angles_l = np.linspace(0, 2*np.pi, n_points_l, endpoint=False)
points_l = np.array([0, 0]) + 5.0 * np.column_stack([np.cos(angles_l), np.sin(angles_l)])
points_l += np.random.normal(0, 0.5, (n_points_l, 2))

# x0_lin_c is the center (array of 2)
x0_lin_c, _ = algebraic_least_squares(points_l, 5.0)

# radial_least_squares(points, radius, initial_center=None, ...)
refined_center, _, _ = radial_least_squares(points_l, 5.0, initial_center=x0_lin_c)

plt.figure(figsize=(10, 8))
plt.scatter(points_l[:, 0], points_l[:, 1], c='blue', label='Noisy points', alpha=0.6)
theta = np.linspace(0, 2*np.pi, 200)

plt.plot(x0_lin_c[0] + 5.0*np.cos(theta), x0_lin_c[1] + 5.0*np.sin(theta), 'g--', linewidth=2, label='Algebraic LS Fit')
plt.plot(refined_center[0] + 5.0*np.cos(theta), 
         refined_center[1] + 5.0*np.sin(theta), 'r-', linewidth=2, label='Radial LS Refined')

plt.legend()
plt.title("Circle Fitting: Iterative Radial vs. Linear Least Squares")
plt.axis('equal')
plt.grid(alpha=0.3)
plt.savefig("radial_ls_result.png", dpi=300, bbox_inches='tight')
plt.close()

print("All results generated successfully.")
