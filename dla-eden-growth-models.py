"""
Diffusion-Limited Aggregation (DLA) and Eden Growth Model

Author:
    Mathias Rendón Fernández

Description:
    Simulation and comparison of two stochastic cluster-growth models:
        - Diffusion-Limited Aggregation (DLA)
        - Eden growth model

    The code calculates and compares:
        - Fractal dimension
        - Cluster roughness
        - Fractal-dimension stability as a function of cluster size
"""

# %%

import random

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Simulation Parameters
# ============================================================================

GRID_SIZE = 50                  # Size of the 2D grid (GRID_SIZE x GRID_SIZE)
INITIAL_RADIUS = 10             # Initial radius from which walkers are released
MAX_STEPS = 10_000              # Maximum number of steps per random walker

CENTER = (GRID_SIZE // 2, GRID_SIZE // 2)


# ============================================================================
# DLA Utilities
# ============================================================================

def maximum_cluster_radius(grid):
    """
    Calculate the maximum distance of an occupied site from the cluster center.

    Parameters
    ----------
    grid : numpy.ndarray
        2D grid representing the cluster.

    Returns
    -------
    float
        Maximum radial distance from the center to an occupied site.
    """
    occupied = np.argwhere(grid == 1)

    if len(occupied) == 0:
        return 0.0

    distances = np.sqrt(
        (occupied[:, 0] - CENTER[0]) ** 2
        + (occupied[:, 1] - CENTER[1]) ** 2
    )

    return np.max(distances)


def cluster_roughness(grid):
    """
    Calculate the roughness of a cluster.

    The roughness is defined as the standard deviation of the highest
    occupied site in each occupied column.

    Parameters
    ----------
    grid : numpy.ndarray
        2D grid representing the cluster.

    Returns
    -------
    float
        Cluster roughness.
    """
    heights = []

    for column in range(grid.shape[1]):
        occupied_rows = np.where(grid[:, column] == 1)[0]

        if len(occupied_rows) > 0:
            heights.append(occupied_rows[-1])

    if not heights:
        return 0.0

    heights = np.array(heights)

    return np.sqrt(np.mean((heights - np.mean(heights)) ** 2))


def random_position_on_circle(radius):
    """
    Generate a random position on a circle centered on the cluster.

    Parameters
    ----------
    radius : float
        Circle radius.

    Returns
    -------
    tuple of int
        Grid coordinates of the randomly selected position.
    """
    angle = random.uniform(0, 2 * np.pi)

    x = int(CENTER[0] + radius * np.cos(angle))
    y = int(CENTER[1] + radius * np.sin(angle))

    # Keep the position inside the grid.
    x = max(0, min(x, GRID_SIZE - 1))
    y = max(0, min(y, GRID_SIZE - 1))

    return x, y


def random_walker(grid, release_radius):
    """
    Simulate a single random walker in the DLA model.

    The walker starts on a circle around the cluster and performs a
    random walk until it either:
        - attaches to the cluster,
        - leaves the grid,
        - moves too far away from the cluster, or
        - reaches the maximum number of steps.

    Parameters
    ----------
    grid : numpy.ndarray
        2D grid representing the DLA cluster.
    release_radius : float
        Current radius at which walkers are released.

    Returns
    -------
    tuple
        Updated release radius and a boolean indicating whether the
        walker successfully attached to the cluster.
    """
    x, y = random_position_on_circle(release_radius)

    for _ in range(MAX_STEPS):
        distance_from_center = np.sqrt(
            (x - CENTER[0]) ** 2
            + (y - CENTER[1]) ** 2
        )

        # Use larger steps when the walker is far from the cluster.
        step_size = 2 if distance_from_center > release_radius - 5 else 1

        dx, dy = random.choice([
            (0, step_size),
            (0, -step_size),
            (step_size, 0),
            (-step_size, 0),
        ])

        x += dx
        y += dy

        # Discard walkers that leave the simulation grid.
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return release_radius, False

        # Check whether the walker is adjacent to the cluster.
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        touches_cluster = any(
            0 <= nx < GRID_SIZE
            and 0 <= ny < GRID_SIZE
            and grid[nx, ny] == 1
            for nx, ny in neighbors
        )

        if touches_cluster:
            grid[x, y] = 1

            # Update the release radius dynamically as the cluster grows.
            max_radius = maximum_cluster_radius(grid)
            release_radius = max(
                release_radius,
                int(max_radius) + 5
            )

            return release_radius, True

        # Discard walkers that move too far away from the cluster.
        if distance_from_center > release_radius * 1.5:
            return release_radius, False

    return release_radius, False


# ============================================================================
# Fractal Dimension
# ============================================================================

def fractal_dimension(grid, max_radius):
    """
    Estimate the fractal dimension using box/radial counting.

    The number of occupied sites within radius R is assumed to scale as

        N(R) ~ R^D

    where D is the fractal dimension. A linear fit in log-log space
    provides an estimate of D.

    Parameters
    ----------
    grid : numpy.ndarray
        2D grid representing the cluster.
    max_radius : float
        Maximum radius considered for the scaling analysis.

    Returns
    -------
    float
        Estimated fractal dimension.
    """
    occupied = np.argwhere(grid == 1)

    if len(occupied) < 2 or max_radius <= 1:
        return np.nan

    distances = np.sqrt(
        (occupied[:, 0] - CENTER[0]) ** 2
        + (occupied[:, 1] - CENTER[1]) ** 2
    )

    radii = np.logspace(0, np.log10(max_radius), 20)

    particle_counts = np.array([
        np.count_nonzero(distances <= radius)
        for radius in radii
    ])

    # Remove zero counts to avoid log(0).
    valid = particle_counts > 0

    log_radii = np.log(radii[valid])
    log_counts = np.log(particle_counts[valid])

    slope, _ = np.polyfit(log_radii, log_counts, 1)

    return slope


# ============================================================================
# DLA Simulation
# ============================================================================

def simulate_dla(num_walkers):
    """
    Simulate a Diffusion-Limited Aggregation cluster.

    Parameters
    ----------
    num_walkers : int
        Number of random walkers released.

    Returns
    -------
    tuple
        Final cluster grid, estimated fractal dimension, and cluster roughness.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[CENTER] = 1

    release_radius = INITIAL_RADIUS

    for _ in range(num_walkers):
        release_radius, _ = random_walker(
            grid,
            release_radius
        )

    dimension = fractal_dimension(grid, release_radius)
    roughness = cluster_roughness(grid)

    return grid, dimension, roughness


# ============================================================================
# Eden Growth Model
# ============================================================================

def simulate_eden(num_growth_steps):
    """
    Simulate an Eden growth cluster.

    At each step, an occupied site is selected randomly and one of its
    neighboring sites is randomly chosen. If the selected neighboring
    site is empty, it becomes occupied.

    Parameters
    ----------
    num_growth_steps : int
        Number of growth attempts.

    Returns
    -------
    tuple
        Final cluster grid, estimated fractal dimension, and cluster roughness.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[CENTER] = 1

    occupied_sites = [CENTER]

    for _ in range(num_growth_steps):
        x, y = random.choice(occupied_sites)

        dx, dy = random.choice([
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ])

        new_x, new_y = x + dx, y + dy

        if (
            0 <= new_x < GRID_SIZE
            and 0 <= new_y < GRID_SIZE
            and grid[new_x, new_y] == 0
        ):
            grid[new_x, new_y] = 1
            occupied_sites.append((new_x, new_y))

    max_radius = maximum_cluster_radius(grid)

    dimension = fractal_dimension(grid, max_radius)
    roughness = cluster_roughness(grid)

    return grid, dimension, roughness


# ============================================================================
# Main Simulation
# ============================================================================

NUM_WALKERS = [20, 50, 100, 200, 500, 1000]

dla_dimensions = []
eden_dimensions = []

# Simulate DLA clusters with increasing numbers of walkers.
for num_walkers in NUM_WALKERS:
    _, dimension, _ = simulate_dla(num_walkers)
    dla_dimensions.append(dimension)

# Simulate Eden clusters with increasing numbers of growth attempts.
for num_walkers in NUM_WALKERS:
    _, dimension, _ = simulate_eden(num_walkers)
    eden_dimensions.append(dimension)

# Generate final clusters for visualization and roughness analysis.
dla_grid, dla_dimension, dla_roughness = simulate_dla(NUM_WALKERS[-1])
eden_grid, eden_dimension, eden_roughness = simulate_eden(NUM_WALKERS[-1])


# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(
    dla_grid,
    cmap="Greys",
    origin="lower"
)
axes[0].set_title("DLA Cluster")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].imshow(
    eden_grid,
    cmap="Greys",
    origin="lower"
)
axes[1].set_title("Eden Cluster")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
plt.show()


# Fractal dimension convergence.
plt.figure(figsize=(8, 5))

plt.plot(
    NUM_WALKERS,
    dla_dimensions,
    "o-",
    label="DLA"
)

plt.plot(
    NUM_WALKERS,
    eden_dimensions,
    "o-",
    label="Eden"
)

plt.xlabel("Number of growth events")
plt.ylabel("Fractal dimension")
plt.title("Fractal Dimension Convergence")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================================
# Results
# ============================================================================

print(
    f"\nFractal dimension of the DLA cluster "
    f"({NUM_WALKERS[-1]} walkers): {dla_dimension:.4f}"
)

print(
    f"Fractal dimension of the Eden cluster "
    f"({NUM_WALKERS[-1]} growth events): {eden_dimension:.4f}"
)

print(f"\nDLA cluster roughness: {dla_roughness:.4f}")
print(f"Eden cluster roughness: {eden_roughness:.4f}")

# %%