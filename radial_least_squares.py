import numpy as np
import matplotlib.pyplot as plt

def radial_least_squares(points, radius, initial_center=None, max_iter=100, tol=1e-6, lambda_damping=0.01):
    """
    Fit a circle with known radius to points using Levenberg-Marquardt algorithm.
    
    Parameters:
    -----------
    points : ndarray, shape (N, 2)
        Array of 2D points [x, y]
    radius : float
        Known radius of the circle
    initial_center : ndarray, shape (2,), optional
        Initial guess for center. If None, uses centroid of points
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    lambda_damping : float
        Damping parameter for Levenberg-Marquardt
    
    Returns:
    --------
    center : ndarray, shape (2,)
        Fitted center [x0, y0]
    iterations : int
        Number of iterations performed
    residuals : ndarray
        Final residuals for each point
    """
    
    # Step 1: Initialize center
    if initial_center is None:
        center = np.mean(points, axis=0)  # Centroid
    else:
        center = np.array(initial_center, dtype=float)
    
    N = len(points)
    
    for iteration in range(max_iter):
        # Step 2: Compute residuals
        # Distance from each point to current center
        diff = points - center  # Shape: (N, 2)
        distances = np.sqrt(np.sum(diff**2, axis=1))  # Shape: (N,)
        
        # Residuals: r_i = d_i - R
        residuals = distances - radius  # Shape: (N,)
        
        # Step 3: Compute Jacobian matrix
        # J[i, 0] = ∂r_i/∂x0 = (x0 - x_i) / d_i
        # J[i, 1] = ∂r_i/∂y0 = (y0 - y_i) / d_i
        
        # Avoid division by zero
        distances_safe = np.where(distances < 1e-10, 1e-10, distances)
        
        J = np.zeros((N, 2))
        J[:, 0] = (center[0] - points[:, 0]) / distances_safe  # ∂r/∂x0
        J[:, 1] = (center[1] - points[:, 1]) / distances_safe  # ∂r/∂y0
        
        # Step 4: Update center using Levenberg-Marquardt
        # c^(k+1) = c^(k) - (J^T J + λI)^(-1) J^T r
        
        JtJ = J.T @ J  # Shape: (2, 2)
        Jtr = J.T @ residuals  # Shape: (2,)
        
        # Add damping term
        damped_matrix = JtJ + lambda_damping * np.eye(2)
        
        # Compute update step
        try:
            update = np.linalg.solve(damped_matrix, Jtr)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {iteration}")
            break
        
        # Update center
        new_center = center - update
        
        # Step 5: Check convergence
        center_change = np.linalg.norm(new_center - center)
        
        center = new_center
        
        if center_change < tol:
            print(f"Converged in {iteration + 1} iterations")
            break
    else:
        print(f"Reached maximum iterations ({max_iter})")
    
    # Compute final residuals
    diff = points - center
    distances = np.sqrt(np.sum(diff**2, axis=1))
    residuals = distances - radius
    
    return center, iteration + 1, residuals


def plot_fit(points, radius, center):
    """
    Visualize the fitted circle and points.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Data points', zorder=3)
    
    # Plot fitted circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    plt.plot(circle_x, circle_y, 'r-', linewidth=2, label='Fitted circle')
    
    # Plot center
    plt.scatter(center[0], center[1], c='red', s=200, marker='x', linewidths=3, label='Center', zorder=4)
    
    # Draw radii to show fit
    for i in range(len(points)):
        plt.plot([center[0], points[i, 0]], [center[1], points[i, 1]], 'g--', alpha=0.3, linewidth=1)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Radial Least Squares Fit\nCenter: ({center[0]:.3f}, {center[1]:.3f}), Radius: {radius:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    # Example 1: Perfect circle with noise
    print("=" * 50)
    print("Example 1: Noisy circle data")
    print("=" * 50)
    
    # Generate synthetic data: circle with center (2, 3), radius 5
    true_center = np.array([2.0, 3.0])
    true_radius = 5.0
    
    # Generate points on circle with noise
    np.random.seed(42)
    n_points = 20
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Perfect circle points
    perfect_points = true_center + true_radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.2, (n_points, 2))
    noisy_points = perfect_points + noise
    
    # Fit circle
    fitted_center, iterations, residuals = radial_least_squares(
        noisy_points, 
        true_radius,
        max_iter=100,
        tol=1e-6
    )
    
    print(f"True center:   ({true_center[0]:.3f}, {true_center[1]:.3f})")
    print(f"Fitted center: ({fitted_center[0]:.3f}, {fitted_center[1]:.3f})")
    print(f"Iterations: {iterations}")
    print(f"RMS error: {np.sqrt(np.mean(residuals**2)):.6f}")
    
    # Visualize
    plot_fit(noisy_points, true_radius, fitted_center)
    
    # Example 2: Simple 4-point example
    print("\n" + "=" * 50)
    print("Example 2: Simple 4-point case")
    print("=" * 50)
    
    simple_points = np.array([
        [3.0, 0.0],
        [0.0, 3.0],
        [-3.0, 0.0],
        [0.0, -3.0]
    ])
    
    fitted_center2, iterations2, residuals2 = radial_least_squares(
        simple_points,
        radius=3.0,
        initial_center=np.array([0.5, 0.5])  # Intentionally offset
    )
    
    print(f"Fitted center: ({fitted_center2[0]:.6f}, {fitted_center2[1]:.6f})")
    print(f"Expected: (0.0, 0.0)")
    print(f"Iterations: {iterations2}")
    
    plot_fit(simple_points, 3.0, fitted_center2)