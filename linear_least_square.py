import numpy as np
import matplotlib.pyplot as plt

def algebraic_least_squares(points, radius):
    """
    Fit circle with known radius using algebraic least squares OPTIMIZATION.
    
    Minimizes: J(x0, y0) = Σ[(xi-x0)² + (yi-y0)² - R²]²
    
    This is reformulated as a linear least squares problem:
    minimize ||Az - b||²
    
    Solution: z* = (AᵀA)⁻¹Aᵀb (optimal closed-form)
    """
    
    N = len(points)
    
    # Step 1: Build matrix A (N x 3)
    # Each row: [1, -2*xi, -2*yi]
    A = np.zeros((N, 3))
    A[:, 0] = 1                    # First column: all ones
    A[:, 1] = -2 * points[:, 0]    # Second column: -2*xi
    A[:, 2] = -2 * points[:, 1]    # Third column: -2*yi
    
    # Step 2: Build vector b (N x 1)
    # Each element: -(xi^2 + yi^2)
    b = -(points[:, 0]**2 + points[:, 1]**2)
    
    # Step 3: Solve the linear system Az = b using least squares
    # z = (A^T A)^(-1) A^T b
    # This is done automatically by np.linalg.lstsq
    z, residuals_lstsq, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # After computing z, change to:
    cost = np.sum((A @ z - b)**2)
    print(f"\n--- Optimization Results ---")
    print(f"Objective J(z*) minimized to: {cost:.6e}")
    print(f"Number of equations (N): {N}")
    print(f"Number of unknowns: 3 (k, x0, y0)")
    
    # Alternative manual computation:
    # z = np.linalg.inv(A.T @ A) @ A.T @ b
    
    # Step 4: Extract solution
    # z = [k, x0, y0]
    k = z[0]
    x0 = z[1]
    y0 = z[2]
    
    center = np.array([x0, y0])
    
    # Verify: k should equal x0^2 + y0^2 - R^2
    k_expected = x0**2 + y0**2 - radius**2
    print(f"k from solution: {k:.6f}")
    print(f"k expected (x0² + y0² - R²): {k_expected:.6f}")
    print(f"Difference: {abs(k - k_expected):.6e}")
    
    # Compute residuals (geometric error for comparison)
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    residuals = distances - radius
    
    return center, residuals


def plot_fit(points, radius, center, title="Algebraic Least Squares Fit"):
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
    plt.title(f'{title}\nCenter: ({center[0]:.3f}, {center[1]:.3f}), Radius: {radius:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def compare_methods(points, radius):
    """
    Compare algebraic (non-iterative) vs radial (iterative) methods.
    """
    print("\n" + "="*60)
    print("COMPARISON: Algebraic LS vs Radial LS")
    print("="*60)
    
    # Algebraic LS
    print("\n--- Algebraic Least Squares (Non-Iterative) ---")
    center_alg, residuals_alg = algebraic_least_squares(points, radius)
    print(f"Center: ({center_alg[0]:.6f}, {center_alg[1]:.6f})")
    print(f"RMS error: {np.sqrt(np.mean(residuals_alg**2)):.6f}")
    
    # For comparison, you could import and run radial LS here
    # (assuming you have the radial_least_squares function from previous code)
    
    return center_alg, residuals_alg


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    

    
    
    # Example 1: Minimal case (N=3, exactly determined)
    print("\n" + "="*60)
    print("Example 1: Minimal case (N=3, exactly determined)")
    print("="*60)
    
    minimal_points = np.array([
        [5.0, 0.0],
        [0.0, 5.0],
        [-5.0, 0.0]
    ])
    
    center3, residuals3 = algebraic_least_squares(minimal_points, radius=5.0)
    print(f"\nFitted center: ({center3[0]:.6f}, {center3[1]:.6f})")
    print(f"Expected center: (0.0, 0.0)")
    print(f"RMS error: {np.sqrt(np.mean(residuals3**2)):.10f}")
    
    plot_fit(minimal_points, 5.0, center3, "Example 1: Minimal case (N=3)")
    
    
    # Example 2: Show it works for over-determined case
    print("\n" + "="*60)
    print("Example 2: Large over-determined case (N=100)")
    print("="*60)
    
    # Many points with small noise
    n_points_large = 100
    angles_large = np.linspace(0, 2*np.pi, n_points_large, endpoint=False)
    true_center_large = np.array([1.5, -2.5])
    true_radius_large = 7.0
    
    perfect_large = true_center_large + true_radius_large * np.column_stack([
        np.cos(angles_large),
        np.sin(angles_large)
    ])
    
    noise_large = np.random.normal(0, 0.1, (n_points_large, 2))
    noisy_large = perfect_large + noise_large
    
    center4, residuals4 = algebraic_least_squares(noisy_large, true_radius_large)
    
    print(f"\nTrue center:   ({true_center_large[0]:.3f}, {true_center_large[1]:.3f})")
    print(f"Fitted center: ({center4[0]:.3f}, {center4[1]:.3f})")
    print(f"Error: {np.linalg.norm(center4 - true_center_large):.6f}")
    print(f"RMS residual: {np.sqrt(np.mean(residuals4**2)):.6f}")
    
    plot_fit(noisy_large, true_radius_large, center4, "Example 2: 100 points")