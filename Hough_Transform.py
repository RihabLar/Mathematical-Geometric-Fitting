import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def hough_circle_known_radius(points, radius, grid_resolution=0.2, search_margin=1.5, 
                               n_angle_samples=360, use_gaussian_smoothing=True, sigma=2.0):
    """
    Fit a circle with known radius using Hough Transform (voting method) - FIXED VERSION.
    
    This version uses:
    - Coarser grid resolution for better vote accumulation
    - More angle samples for complete coverage
    - Gaussian smoothing to create smooth peaks
    - Smaller search space to concentrate votes
    
    Parameters:
    -----------
    points : ndarray, shape (N, 2)
        Array of 2D points [x, y]
    radius : float
        Known radius of the circle
    grid_resolution : float
        Grid spacing for accumulator (0.2-0.5 recommended)
    search_margin : float
        Margin around data bounds for search space (in units of radius)
    n_angle_samples : int
        Number of angles to sample around each point (360 recommended)
    use_gaussian_smoothing : bool
        Apply Gaussian smoothing to accumulator
    sigma : float
        Standard deviation for Gaussian smoothing
    
    Returns:
    --------
    center : ndarray, shape (2,)
        Fitted center [x0, y0] (peak in accumulator)
    accumulator : ndarray
        The vote accumulator array (smoothed if enabled)
    x_bins : ndarray
        X-axis bin edges
    y_bins : ndarray
        Y-axis bin edges
    votes : float
        Vote value at peak
    """
    
    N = len(points)
    
    # Step 1: Define search space (accumulator bounds)
    x_min = np.min(points[:, 0]) - search_margin * radius
    x_max = np.max(points[:, 0]) + search_margin * radius
    y_min = np.min(points[:, 1]) - search_margin * radius
    y_max = np.max(points[:, 1]) + search_margin * radius
    
    # Create bin edges
    x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    
    # Initialize accumulator
    accumulator = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    
    print(f"Accumulator size: {accumulator.shape}")
    print(f"Grid resolution: {grid_resolution}")
    print(f"Search space: x ∈ [{x_min:.2f}, {x_max:.2f}], y ∈ [{y_min:.2f}, {y_max:.2f}]")
    
    # Step 2: Voting process with SOFT VOTING (Gaussian weights)
    angles = np.linspace(0, 2*np.pi, n_angle_samples, endpoint=False)
    
    print(f"Starting voting with {N} points, {n_angle_samples} angles, soft voting...")
    
    # Precompute for efficiency
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    for i, point in enumerate(points):
        xi, yi = point
        
        # Compute ALL candidate centers at once (vectorized)
        candidate_x = xi + radius * cos_angles
        candidate_y = yi + radius * sin_angles
        
        # Convert to bin indices
        x_indices = ((candidate_x - x_min) / grid_resolution).astype(int)
        y_indices = ((candidate_y - y_min) / grid_resolution).astype(int)
        
        # Soft voting: spread vote to nearby cells with Gaussian weights
        for x_idx, y_idx in zip(x_indices, y_indices):
            # Check bounds
            if 0 <= x_idx < accumulator.shape[1] and 0 <= y_idx < accumulator.shape[0]:
                # Vote in a 5x5 neighborhood with Gaussian weights
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny = y_idx + dy
                        nx = x_idx + dx
                        
                        if 0 <= ny < accumulator.shape[0] and 0 <= nx < accumulator.shape[1]:
                            # Gaussian weight based on distance from center
                            weight = np.exp(-0.5 * (dx**2 + dy**2) / (sigma**2))
                            accumulator[ny, nx] += weight
    
    # Step 3: Optional Gaussian smoothing for even smoother peaks
    if use_gaussian_smoothing:
        accumulator = gaussian_filter(accumulator, sigma=sigma)
        print("Applied Gaussian smoothing to accumulator")
    
    # Step 4: Find peak in accumulator
    peak_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    y_peak_idx, x_peak_idx = peak_idx
    
    # Convert bin index to coordinate (use center of bin)
    x0 = x_min + (x_peak_idx + 0.5) * grid_resolution
    y0 = y_min + (y_peak_idx + 0.5) * grid_resolution
    
    center = np.array([x0, y0])
    votes = accumulator[y_peak_idx, x_peak_idx]
    
    print(f"Peak found at: ({x0:.6f}, {y0:.6f}) with vote value: {votes:.2f}")
    
    # Optional: Sub-pixel refinement using weighted centroid around peak
    refinement_radius = 2
    y_start = max(0, y_peak_idx - refinement_radius)
    y_end = min(accumulator.shape[0], y_peak_idx + refinement_radius + 1)
    x_start = max(0, x_peak_idx - refinement_radius)
    x_end = min(accumulator.shape[1], x_peak_idx + refinement_radius + 1)
    
    local_acc = accumulator[y_start:y_end, x_start:x_end]
    
    if np.sum(local_acc) > 0:
        y_coords = np.arange(y_start, y_end)
        x_coords = np.arange(x_start, x_end)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        
        x_refined = np.sum(X * local_acc) / np.sum(local_acc)
        y_refined = np.sum(Y * local_acc) / np.sum(local_acc)
        
        x0_refined = x_min + (x_refined + 0.5) * grid_resolution
        y0_refined = y_min + (y_refined + 0.5) * grid_resolution
        
        center = np.array([x0_refined, y0_refined])
        print(f"After sub-pixel refinement: ({x0_refined:.6f}, {y0_refined:.6f})")
    
    return center, accumulator, x_bins, y_bins, votes


def plot_hough_results(points, radius, center, accumulator, x_bins, y_bins, votes):
    """
    Visualize Hough Transform results: fitted circle and accumulator heatmap.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Left plot: Fitted circle ---
    ax1.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Data points', zorder=3)
    
    # Plot fitted circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'r-', linewidth=2, label='Fitted circle')
    
    # Plot center
    ax1.scatter(center[0], center[1], c='red', s=200, marker='x', linewidths=3, 
                label=f'Center (votes={votes:.1f})', zorder=4)
    
    # Draw radii
    for i in range(len(points)):
        ax1.plot([center[0], points[i, 0]], [center[1], points[i, 1]], 
                'g--', alpha=0.3, linewidth=1)
    
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'Hough Transform Fit\nCenter: ({center[0]:.3f}, {center[1]:.3f}), Radius: {radius:.3f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # --- Right plot: Accumulator heatmap ---
    im = ax2.imshow(accumulator, origin='lower', 
                    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                    cmap='hot', aspect='auto', interpolation='bilinear')
    
    # Mark the peak
    ax2.scatter(center[0], center[1], c='cyan', s=300, marker='*', 
                edgecolors='white', linewidths=2, label='Peak (fitted center)', zorder=5)
    
    # Plot data points on accumulator
    ax2.scatter(points[:, 0], points[:, 1], c='lime', s=30, marker='o',
                edgecolors='white', linewidths=1, label='Data points', zorder=4, alpha=0.7)
    
    plt.colorbar(im, ax=ax2, label='Vote Intensity')
    ax2.set_title(f'Hough Accumulator (Parameter Space)\nPeak intensity: {votes:.1f}')
    ax2.set_xlabel('x₀ (center x-coordinate)')
    ax2.set_ylabel('y₀ (center y-coordinate)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()




# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    
    
    # Example1: noisy data
    print("\n" + "="*70)
    print("Example 1: Circle with 20 points (moderate noise)")
    print("="*70)
    
    true_center2 = np.array([0.0, 0.0])
    true_radius2 = 5.0
    
    n_points2 = 20
    angles2 = np.linspace(0, 2*np.pi, n_points2, endpoint=False)
    points2 = true_center2 + true_radius2 * np.column_stack([
        np.cos(angles2),
        np.sin(angles2)
    ])
    points2 += np.random.normal(0, 0.3, (n_points2, 2))
    
    center2, acc2, x_bins2, y_bins2, votes2 = hough_circle_known_radius(
        points2, true_radius2,
        grid_resolution=0.2,
        n_angle_samples=360
    )
    
    print(f"\nTrue center: ({true_center2[0]:.3f}, {true_center2[1]:.3f})")
    print(f"Fitted center: ({center2[0]:.3f}, {center2[1]:.3f})")
    print(f"Error: {np.linalg.norm(center2 - true_center2):.6f}")
    
    distances2 = np.sqrt(np.sum((points2 - center2)**2, axis=1))
    residuals2 = distances2 - true_radius2
    print(f"RMS error: {np.sqrt(np.mean(residuals2**2)):.6f}")
    
    plot_hough_results(points2, true_radius2, center2, acc2, x_bins2, y_bins2, votes2)
    
    
    