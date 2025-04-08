import goo
import numpy as np

goo.reset_modules()
goo.reset_scene()


def random_point_inside_sphere(radius):
    """Generate a random point inside a sphere."""
    r = (radius - 1) * np.cbrt(np.random.rand())
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def check_min_distance(new_point, points, min_distance):
    """Check if the new point meets the minimum distance requirement."""
    for point in points:
        distance = np.linalg.norm(np.array(new_point) - np.array(point))
        if distance < min_distance:
            return False
    return True


# Create cells within the sphere with minimum distance constraint
num_cells = 8  # Number of cells
radius = 15  # Sphere radius
min_distance = 3  # Minimum distance between cells

goo.create_boundary((0, 0, 0), size=radius * 1.2)

cellsA = goo.CellType("cellA", target_volume=70, pattern="simple")
cellsA.homo_adhesion_strength = 250
cells = []

while len(cells) < num_cells:
    new_point = random_point_inside_sphere(radius)
    if check_min_distance(new_point, cells, min_distance):
        cells.append(new_point)
        cell_name = f"cellA{len(cells)}"
        color = tuple(np.random.random_sample(3))
        cell = cellsA.create_cell(cell_name, new_point, color=color)
        cell.stiffness = 5

sim = goo.Simulator(celltypes=[cellsA], time=350, physics_dt=1)
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=60, sigma=2),
        goo.RecenterHandler(),
        goo.RandomMotionHandler(distribution=goo.ForceDist.GAUSSIAN, strength=2500),
    ]
)
