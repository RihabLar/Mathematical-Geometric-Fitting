import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

def ransac_circle_known_radius(points, radius, threshold=0.3, iterations=1000, 
                                refine=True, random_seed=None):
    """
    Fit a circle with known radius using RANSAC (iterative robust method).
    
    RANSAC is robust to outliers by randomly sampling minimal sets and
    finding the model with maximum consensus (most inliers).
    
    Parameters:
    -----------
    points : ndarray, shape (N, 2)
        Array of 2D points [x, y]
    radius : float
        Known radius of the circle
    threshold : float
        Distance threshold for inlier classification (default: 0.3)
    iterations : int
        Number of RANSAC iterations (default: 1000)
    refine : bool
        Whether to refine result using all inliers with Radial LS (default: True)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    center : ndarray, shape (2,)
        Fitted center [x0, y0]
    inliers : ndarray, shape (M,)
        Boolean array indicating which points are inliers
    best_score : int
        Number of inliers in best model
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N = len(points)
    
    if N < 2:
        raise ValueError("Need at least 2 points")
    
    print(f"RANSAC: {N} points, R={radius}, τ={threshold}, {iterations} iterations")
    
    best_center = None
    best_inliers = None
    best_score = 0
    
    # RANSAC main loop
    for iteration in range(iterations):
        # Step 1: SAMPLE - randomly select 2 points
        sample_indices = np.random.choice(N, size=2, replace=False)
        p1 = points[sample_indices[0]]
        p2 = points[sample_indices[1]]
        
        # Step 2: FIT - compute candidate center(s) from 2 points
        candidate_centers = compute_centers_from_two_points(p1, p2, radius)
        
        if candidate_centers is None:
            continue  # Invalid sample (points too far apart)
        
        # Step 3: EVALUATE - test each candidate center
        for candidate_center in candidate_centers:
            # Compute distances from all points to candidate center
            distances = np.sqrt(np.sum((points - candidate_center)**2, axis=1))
            
            # Count inliers: points where |distance - R| < threshold
            residuals = np.abs(distances - radius)
            inliers = residuals < threshold
            score = np.sum(inliers)
            
            # Step 4: UPDATE - keep best model
            if score > best_score:
                best_score = score
                best_center = candidate_center
                best_inliers = inliers
    
    if best_center is None:
        raise RuntimeError("RANSAC failed to find any valid model")
    
    inlier_ratio = best_score / N
    print(f"Best model: {best_score}/{N} inliers ({inlier_ratio*100:.1f}%)")
    print(f"Initial center: ({best_center[0]:.6f}, {best_center[1]:.6f})")
    
    # Step 5: REFINE - fit model to all inliers using Radial LS
    if refine and best_score >= 3:
        inlier_points = points[best_inliers]
        refined_center = refine_with_radial_ls(inlier_points, radius, 
                                                initial_center=best_center)
        print(f"Refined center: ({refined_center[0]:.6f}, {refined_center[1]:.6f})")
        return refined_center, best_inliers, best_score
    
    return best_center, best_inliers, best_score


def compute_centers_from_two_points(p1, p2, radius):
    """
    Compute possible circle centers given two points on the circle with known radius.
    
    Returns None, 1, or 2 centers depending on point separation.
    
    Parameters:
    -----------
    p1, p2 : ndarray, shape (2,)
        Two points on the circle
    radius : float
        Known radius
    
    Returns:
    --------
    centers : list of ndarray or None
        List of candidate centers (0, 1, or 2 centers)
    """
    
    # Distance between the two points
    d = np.linalg.norm(p2 - p1)
    
    # Case 1: Points too far apart (d > 2R)
    if d > 2 * radius + 1e-6:
        return None  # No valid circle
    
    # Case 2: Points are diametrically opposite (d ≈ 2R)
    if np.abs(d - 2 * radius) < 1e-6:
        center = (p1 + p2) / 2
        return [center]
    
    # Case 3: Two intersection points (d < 2R)
    # Midpoint
    m = (p1 + p2) / 2
    
    # Unit vector along p1 to p2
    u = (p2 - p1) / d
    
    # Perpendicular vector
    v = np.array([-u[1], u[0]])
    
    # Distance from midpoint to centers
    h = np.sqrt(radius**2 - (d/2)**2)
    
    # Two candidate centers
    c1 = m + h * v
    c2 = m - h * v
    
    return [c1, c2]


def refine_with_radial_ls(points, radius, initial_center, max_iter=50, tol=1e-6):
    """
    Refine center estimate using Radial Least Squares (Levenberg-Marquardt).
    
    Minimizes: Σ(||pi - c|| - R)²
    
    Parameters:
    -----------
    points : ndarray, shape (N, 2)
        Inlier points
    radius : float
        Known radius
    initial_center : ndarray, shape (2,)
        Initial guess for center
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    center : ndarray, shape (2,)
        Refined center
    """
    
    center = initial_center.copy()
    lambda_damping = 0.01
    
    for iteration in range(max_iter):
        # Compute distances and residuals
        diff = points - center
        distances = np.sqrt(np.sum(diff**2, axis=1))
        residuals = distances - radius
        
        # Avoid division by zero
        distances_safe = np.where(distances < 1e-10, 1e-10, distances)
        
        # Jacobian
        J = np.zeros((len(points), 2))
        J[:, 0] = (center[0] - points[:, 0]) / distances_safe
        J[:, 1] = (center[1] - points[:, 1]) / distances_safe
        
        # Levenberg-Marquardt update
        JtJ = J.T @ J
        Jtr = J.T @ residuals
        
        try:
            update = np.linalg.solve(JtJ + lambda_damping * np.eye(2), Jtr)
        except np.linalg.LinAlgError:
            break
        
        new_center = center - update
        
        # Check convergence
        if np.linalg.norm(new_center - center) < tol:
            break
        
        center = new_center
    
    return center


def plot_ransac_results(points, radius, center, inliers, title="RANSAC Circle Fit"):
    """
    Visualize RANSAC results showing inliers and outliers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Separate inliers and outliers
    inlier_points = points[inliers]
    outlier_points = points[~inliers]
    
    # --- Left plot: Fitted circle with inliers/outliers ---
    if len(inlier_points) > 0:
        ax1.scatter(inlier_points[:, 0], inlier_points[:, 1], 
                   c='blue', s=50, label=f'Inliers ({len(inlier_points)})', zorder=3)
    
    if len(outlier_points) > 0:
        ax1.scatter(outlier_points[:, 0], outlier_points[:, 1], 
                   c='red', s=100, marker='x', linewidths=2,
                   label=f'Outliers ({len(outlier_points)})', zorder=3)
    
    # Plot fitted circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'g-', linewidth=2, label='Fitted circle')
    
    # Plot center
    ax1.scatter(center[0], center[1], c='green', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='Center', zorder=4)
    
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'{title}\nCenter: ({center[0]:.3f}, {center[1]:.3f}), Radius: {radius:.3f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # --- Right plot: Residuals ---
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    residuals = np.abs(distances - radius)
    
    colors = ['blue' if inlier else 'red' for inlier in inliers]
    ax2.bar(range(len(points)), residuals, color=colors, alpha=0.7)
    ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, 
                label='Inlier threshold')
    ax2.set_xlabel('Point index')
    ax2.set_ylabel('|Distance to center - R|')
    ax2.set_title('Residuals (Blue=Inlier, Red=Outlier)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    # Example 1: Extreme case - 70% outliers (near RANSAC limit)
    print("\n" + "="*70)
    print("Example 1: Extreme case with 70% outliers")
    print("="*70)
    
    true_center5 = np.array([2.5, 2.5])
    true_radius5 = 4.5
    
    # Inliers (30% - minimum for RANSAC to work)
    n_inliers5 = 12
    angles5 = np.linspace(0, 2*np.pi, n_inliers5, endpoint=False)
    inliers5 = true_center5 + true_radius5 * np.column_stack([
        np.cos(angles5),
        np.sin(angles5)
    ])
    inliers5 += np.random.normal(0, 0.2, (n_inliers5, 2))
    
    # Outliers (70%)
    n_outliers5 = 28
    outliers5 = np.random.uniform(-4, 10, (n_outliers5, 2))
    
    points5 = np.vstack([inliers5, outliers5])
    
    print("This is challenging! RANSAC needs many iterations...")
    center5, inliers_mask5, score5 = ransac_circle_known_radius(
        points5, true_radius5, threshold=0.4, iterations=5000, refine=True, random_seed=42
    )
    
    print(f"\nTrue center: ({true_center5[0]:.3f}, {true_center5[1]:.3f})")
    print(f"Fitted center: ({center5[0]:.3f}, {center5[1]:.3f})")
    print(f"Error: {np.linalg.norm(center5 - true_center5):.6f}")
    
    if np.linalg.norm(center5 - true_center5) < 0.5:
        print("✓ RANSAC succeeded despite 70% outliers!")
    else:
        print("✗ RANSAC struggled with this extreme outlier ratio")
    
    plot_ransac_results(points5, true_radius5, center5, inliers_mask5,
                       "Example 5: Extreme 70% Outliers")
    
    
