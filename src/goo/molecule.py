from typing import Optional, Tuple
from typing_extensions import override
import numpy as np
import math

from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
from scipy.spatial import KDTree

from goo.utils import *


class Molecule:
    """A molecule involved in the diffusion system.

    Args:
        name (str): The name of the molecule.
        conc (float): The initial concentration of the molecule.
        D (float): The diffusion rate of the molecule.
        gradient (str, optional): The gradient of the molecule. Defaults to None.
    """

    def __init__(
        self, name: str, conc: float, D: float, gradient: Optional[str] = None
    ):
        self.name = name
        self.D = D
        self.conc = conc
        self.gradient = gradient

    def __repr__(self):
        str = f"Molecule(concentration={self.conc}" f"diffusion_rate={self.D})"
        return str

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hash(self) == hash(other)


class DiffusionSystem:
    """A diffusion system that simulates the diffusion of molecules in a 3D grid.
    It also handles the secretion and sensing of molecular signals by cells.

    Args:
        molecules (list[Molecule]): The list of molecules in the system.
        grid_size (Tuple[int, int, int], optional): The size of the 3D grid. Defaults to (25, 25, 25).
        grid_center (Tuple[int, int, int], optional): The center of the 3D grid. Defaults to (0, 0, 0).
        time_step (float, optional): The time step of the simulation. Defaults to 0.1.
        total_time (int, optional): The total time of the simulation. Defaults to 10.
        element_size (Tuple[float, float, float], optional): The size of each element in the grid. Defaults to (1.0, 1.0, 1.0).
    """

    def __init__(
        self,
        molecules: list[Molecule],
        grid_size: Tuple[int, int, int] = (50, 50, 50),
        grid_center: Tuple[int, int, int] = (0, 0, 0),
        time_step: float = 0.1,
        total_time: int = 1,
        element_size=(0.5, 0.5, 0.5),
    ) -> None:
        self.molecules = molecules
        self.grid_size = grid_size
        self.time_step = time_step
        self.total_time = total_time
        self.grid_center = grid_center
        self.element_size = element_size

        self._grid_concentrations = {}
        self._kd_tree = None

    def build_kdtree(self):
        """Initialize the grid and build its corresponding KD-Tree."""
        xgrid, ygrid, zgrid = self.grid_size
        xlim, ylim, zlim = (np.array(self.grid_size) - 1) / 2 * self.element_size

        # Create x, y, z coordinates centered around 0 
        # with specified range and space between grid points.
        x, y, z = np.mgrid[
            -xlim : xlim : complex(xgrid),
            -ylim : ylim : complex(ygrid),
            -zlim : zlim : complex(zgrid),
        ]
        grid_points = np.c_[x.ravel(), y.ravel(), z.ravel()]
        grid_points += np.array(self.grid_center)
        self._kd_tree = KDTree(grid_points)

        # initialize the grid to store concentrations for each molecule
        for mol in self.molecules:
            self._grid_concentrations[mol] = np.zeros(self.grid_size)
            match mol.gradient:
                case None:
                    continue
                case "constant":
                    self._grid_concentrations[mol].fill(mol.conc)
                case "random":
                    variation = 0.1  # 10% variation
                    noise = np.random.normal(
                        0, variation * mol.conc, size=self.grid_size
                    )
                    rand_conc = mol.conc + noise
                    # conc is always positive
                    rand_conc = np.clip(rand_conc, 0, None)
                    self._grid_concentrations[mol] = rand_conc
                case "center":
                    center_index = self._nearest_idx(self.grid_center)
                    self._grid_concentrations[mol].ravel()[center_index] += mol.conc
                case "linear":
                    x_grad = np.linspace(0, mol.conc, self.grid_size[0]).reshape(
                        -1, 1, 1
                    )

                    self._grid_concentrations[mol] = np.tile(
                        x_grad, (1, self.grid_size[1], self.grid_size[2])
                    )

        # add 8 empty points at the corners of the grid in Blender
        corner_offsets = [
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ]
        for offset in corner_offsets:
            corner_position = (
                self.grid_center[0]
                + offset[0] * (self.grid_size[0] - 1) / 2 * self.element_size[0],
                self.grid_center[1]
                + offset[1] * (self.grid_size[1] - 1) / 2 * self.element_size[1],
                self.grid_center[2]
                + offset[2] * (self.grid_size[2] - 1) / 2 * self.element_size[2],
            )
            bpy.ops.object.empty_add(type="PLAIN_AXES", location=corner_position)

        return self._kd_tree

    def _nearest_idx(self, point):
        """Get the nearest grid index for the given point."""
        return self._kd_tree.query(point)[1]

    def update_concentration(self, mol, point, value):
        """Add molecule value to a certain point at a given voxel in the grid."""
        # Convert flat index to 3D index
        idx = self._nearest_idx(point)
        self._grid_concentrations[mol].ravel()[idx] += value

    def get_concentration(self, mol, point):
        """Add molecule value to a certain point at a given voxel in the grid."""
        # Convert flat index to 3D index
        idx = self._nearest_idx(point)
        return self._grid_concentrations[mol].ravel()[idx]

    def get_ball_concentrations(self, center, radius):
        """Get concentrations of all molecules in a sphere 
        with given center and radius."""
        idxs = self._kd_tree.query_ball_point(center, radius)
        signaling_concs = {}
        for mol, grid_concs in self._grid_concentrations.items():
            signaling_concs[mol] = np.sum(grid_concs.ravel()[idxs])
        return signaling_concs

    def get_coords_concentrations(self, mol, center, radius):
        """Get a list of coordinates and a list of molecule 
        concentration for each coordinate."""
        idxs = self._kd_tree.query_ball_point(center, radius)
        coords = self._kd_tree.data[idxs]
        concs = self._grid_concentrations[mol].ravel()[idxs]
        return coords, concs

    def diffuse(self):
        """Simulate the diffusion of molecules in the grid
        using the Laplace operator."""
        for mol in self.molecules:
            conc = self._grid_concentrations[mol]
            laplacian = laplace(conc, mode="wrap")
            diff_coeff = mol.D
            conc += self.time_step * diff_coeff * laplacian
            conc = np.clip(conc, 0, None)
            self._grid_concentrations[mol] = conc

    def simulate_diffusion(self):
        """Simulate the diffusion of molecules in the grid."""
        tot_time = self.total_time
        t_step = self.time_step
        num_steps = int(tot_time / t_step)
        for _ in range(num_steps):
            self.diffuse()
