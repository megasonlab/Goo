import os

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum, Flag, auto

import bmesh
import bpy
import h5py
import numpy as np
import tifffile
import xarray as xr

from mathutils import Vector
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist, squareform
from typing_extensions import override

from goo.cell import Cell
from goo.gene import Gene
from goo.molecule import DiffusionSystem


def save_tiff(data: np.ndarray, path: str) -> None:
    """Save a numpy array as a TIFF file.

    Args:
        data: The numpy array to save
        path: The path where to save the TIFF file
    """
    tifffile.imwrite(path, data)


class Handler(ABC):
    def setup(
        self,
        get_cells: Callable[[], list[Cell]],
        get_diffsystem: Callable[[], DiffusionSystem],
        dt: float,
    ) -> None:
        """Set up the handler.

        Args:
            get_cells: A function that, when called,
                retrieves the list of cells that may divide.
            dt: The time step for the simulation.
        """
        self.get_cells = get_cells
        self.get_diffsystem = get_diffsystem
        self.dt = dt

    @abstractmethod
    def run(self, scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
        """Run the handler.

        This is the function that gets passed to Blender, to be called
        upon specified events (e.g. post-frame change).

        Args:
            scene: The Blender scene.
            depsgraph: The dependency graph.
        """
        raise NotImplementedError("Subclasses must implement run() method.")


class StopHandler(Handler):
    """Handler for stopping the simulation at the end of the simulation time or when reaching max cells."""

    def __init__(self, max_cells: int | None = None):
        super().__init__()
        self.get_cells = None
        self.get_diffsystem = None
        self.dt = None
        self.max_cells = max_cells

    def setup(
        self,
        get_cells: Callable[[], list[Cell]],
        get_diffsystem: Callable[[], DiffusionSystem],
        dt: float,
    ) -> None:
        """Set up the handler.

        Args:
            get_cells: A function that, when called,
                retrieves the list of cells that may divide.
            dt: The time step for the simulation.
        """
        self.get_cells = get_cells
        self.get_diffsystem = get_diffsystem
        self.dt = dt

    def run(self, scene, depsgraph):
        if not self.get_cells:
            print("Warning: StopHandler not properly initialized. Call setup() first.")
            return

        for cell in self.get_cells():
            # Only update point cache if the cell has a valid cloth modifier
            if hasattr(cell, 'cloth_mod') and cell.cloth_mod is not None:
                cell.cloth_mod.point_cache.frame_end = bpy.context.scene.frame_end

        cell_count = len(self.get_cells())
        frame_str = f"Calculating frame {scene.frame_current}"
        total_length = len(frame_str) + 8
        border_line = "=" * total_length

        print(border_line)
        print(f"=== {frame_str} ===")
        print(border_line)
        print(f"Number of cells: {cell_count}")

        # Check if we've reached either the time limit or cell limit
        should_stop = False
        stop_reason = ""

        if scene.frame_current >= bpy.context.scene.frame_end:
            should_stop = True
            stop_reason = f"Simulation has reached the last frame: {scene.frame_current}"

        if self.max_cells is not None and cell_count >= self.max_cells:
            should_stop = True
            stop_reason = f"Simulation has reached maximum number of cells: {cell_count}"

        if should_stop:
            print(f"{stop_reason}. Stopping.")

            # Handle stopping differently based on whether we're rendering or simulating
            if bpy.context.space_data and bpy.context.space_data.type == 'VIEW_3D':
                # We're in the viewport, can use animation_cancel
                bpy.ops.screen.animation_cancel(restore_frame=True)
            else:
                # We're rendering, set the frame end to current frame
                bpy.context.scene.frame_end = scene.frame_current

            # Freeze all cells
            for cell in self.get_cells():
                # Only try to disable physics and remesh if the cell has physics enabled
                if hasattr(cell, 'physics_enabled') and cell.physics_enabled:
                    cell.disable_physics()
                    cell.remesh()

            # Remove all handlers to prevent further processing
            bpy.app.handlers.frame_change_pre.clear()
            bpy.app.handlers.frame_change_post.clear()


class RemeshHandler(Handler):
    """Handler for remeshing cells at given frequencies.

    Attributes:
        freq (int): Number of frames between remeshes.
        smooth_factor (float): Factor to pass to `bmesh.ops.smooth_vert`.
            Disabled if set to 0.
        voxel_size (float): Factor to pass to `voxel_remesh()`. Disabled if set to 0.
        sphere_factor (float): Factor to pass to Cast to sphere modifier.
            Disabled if set to 0.
    """

    def __init__(self, freq=1, voxel_size=None, smooth_factor=0.1, sphere_factor=0):
        self.freq = freq
        self.voxel_size = voxel_size
        self.smooth_factor = smooth_factor
        self.sphere_factor = sphere_factor

    def run(self, scene, depsgraph):
        if scene.frame_current % self.freq != 0:
            return
        for cell in self.get_cells():
            if not cell.physics_enabled or cell.is_collapsed():
                continue

            # Update mesh and disable physics
            bm = bmesh.new()
            bm.from_mesh(cell.obj_eval.to_mesh())
            cell.disable_physics()
            if self.smooth_factor:
                bmesh.ops.smooth_vert(
                    bm,
                    verts=bm.verts,
                    factor=self.smooth_factor,
                )
            bm.to_mesh(cell.obj.data)
            bm.free()
            cell.recenter()

            if self.voxel_size is not None:
                cell.remesh(self.voxel_size)
                cell.recenter()
            else:
                cell.remesh()
                cell.recenter()

            # Recenter and re-enable physics
            cell.enable_physics()
            cell.cloth_mod.point_cache.frame_start = scene.frame_current


class DiffusionHandler(Handler):
    """Handler for simulating diffusion of a substance in the grid in the scene.

    Args:
        diffusionSystem: The reaction-diffusion system to simulate.
    """

    @override
    def setup(
        self,
        get_cells: Callable[[], list[Cell]],
        get_diffsystems: Callable[[], list[DiffusionSystem]],
        dt,
    ):
        """Build the KD-Tree from the grid coordinates if not already built."""
        super().setup(get_cells, get_diffsystems, dt)
        self.get_diffsystem().build_kdtree()

    def run(self, scene, depsgraph) -> None:
        self.get_diffsystem().simulate_diffusion()


class NetworkHandler(Handler):
    """Handler for gene regulatory networks."""

    def run(self, scene, despgraph):
        for cell in self.get_cells():
            cell.step_grn(dt=self.dt)


class RecenterHandler(Handler):
    """Handler for updating cell origin and location of
    cell-associated adhesion locations every frame."""

    def run(self, scene, depsgraph):
        cells = self.get_cells()

        cell_number = len(cells)
        total_volume = np.sum([cell.volume() for cell in cells])
        average_volume = np.mean([cell.volume() for cell in cells])
        valid_pressures = [
            cell.pressure for cell in cells
            if hasattr(cell, 'cloth_mod') and cell.cloth_mod
            and hasattr(cell.cloth_mod, 'settings')
            and hasattr(cell.cloth_mod.settings, 'uniform_pressure_force')
        ]
        average_pressure = np.mean(valid_pressures) if valid_pressures else 0
        _, sphericities, _, _ = _shape_features(cells)
        average_sphericity = np.mean(sphericities)

        bpy.context.scene.world["Cell#"] = cell_number
        bpy.context.scene.world["Avg Volume"] = average_volume
        bpy.context.scene.world["Avg Pressure"] = average_pressure
        bpy.context.scene.world["Avg Sphericity"] = average_sphericity
        bpy.context.scene.world["Total Volume"] = total_volume

        for cell in self.get_cells():
            cell.recenter()

            # Update adhesion forces if they exist
            if hasattr(cell, 'adhesion_forces'):
                cell_size = cell.major_axis().length() / 2
                for force in cell.adhesion_forces:
                    if not force.enabled():
                        continue
                    force.min_dist = cell_size - 0.4
                    force.max_dist = cell_size + 0.4

            # Update motion force if it exists
            if hasattr(cell, 'motion_force') and cell.motion_force:
                cell.move()

            # Update cloth modifier if it exists
            if hasattr(cell, 'cloth_mod') and cell.cloth_mod:
                cell.cloth_mod.point_cache.frame_end = bpy.context.scene.frame_end


class GrowthPIDHandler(Handler):
    @override
    def run(self, scene, depsgraph):
        for cell in self.get_cells():
            cell.step_growth()


"""Possible distributions of random motion."""
ForceDist = Enum("ForceDist", ["CONSTANT", "UNIFORM", "GAUSSIAN"])


class RandomMotionHandler(Handler):
    """Handler for simulating random cell motion.

    At every frame, the direction of motion is is randomly selected
    from a specified distribution, and the strength is set by the user.

    Attributes:
        distribution (ForceDist): Distribution of random location of motion force.
        strength (int): Strength of the motion force.
        persistence (tuple[float, float, float]): Persistent direction of motion force.
    """

    def __init__(
        self,
        distribution: ForceDist = ForceDist.UNIFORM,
        strength: int = 0,
        persistence: tuple[float, float, float] = (0, 0, 0)
    ):
        self.distribution = distribution
        self.strength = strength
        self.persistence = persistence

    def run(self, scene, depsgraph):
        for cell in self.get_cells():
            if not cell.physics_enabled:
                continue
            if not cell.motion_force.enabled:
                cell.motion_force.enable()

            dir = cell.loc
            match self.distribution:
                case ForceDist.CONSTANT:
                    # persistent motion in a single direction
                    dir = self.persistence
                case ForceDist.UNIFORM:
                    # sampled from continuous uniform distribution bounded [0, 1]
                    dir = Vector(self.persistence) \
                        + Vector(np.random.uniform(low=-1, high=1, size=(3,)))
                case ForceDist.GAUSSIAN:
                    dir = Vector(self.persistence) \
                        + Vector(np.random.normal(loc=0, scale=1, size=(3,)))
                case _:
                    raise ValueError(
                        "Motion noise distribution must be one of UNIFORM or GAUSSIAN."
                    )
            if cell.celltype.motion_strength:
                cell.motion_force.strength = cell.celltype.motion_strength
            else:
                cell.motion_force.strength = self.strength
            # move motion force
            cell.move(dir)
            cell.cloth_mod.point_cache.frame_end = bpy.context.scene.frame_end


"""Possible properties by which cells are colored."""
Colorizer = Enum("Colorizer", ["PRESSURE", "VOLUME", "RANDOM", "GENE"])

"""Color map for the random cell colorizer."""
COLORS = [
    (0.902, 0.490, 0.133),  # Orange
    (0.466, 0.674, 0.188),  # Green
    (0.208, 0.592, 0.560),  # Teal
    (0.121, 0.466, 0.705),  # Blue
    (0.682, 0.780, 0.909),  # Light Blue
    (0.984, 0.502, 0.447),  # Coral
    (0.890, 0.101, 0.109),  # Red
    (0.792, 0.698, 0.839),  # Lavender
    (0.415, 0.239, 0.603),  # Purple
    (0.941, 0.894, 0.259),  # Yellow
    (0.650, 0.337, 0.156),  # Brown
    (0.647, 0.647, 0.647),  # Grey
    (0.529, 0.807, 0.980),  # Sky Blue
    (0.556, 0.929, 0.247),  # Light Green
    (0.749, 0.376, 0.980),  # Violet
    (0.980, 0.745, 0.376),  # Peach
    (0.415, 0.215, 0.235),  # Dark Red
    (0.905, 0.725, 0.725),  # Soft Pink
    (0.282, 0.820, 0.800),  # Aqua
    (0.137, 0.137, 0.137),  # Black
]


class ColorizeHandler(Handler):
    """Handler for coloring cells based on a specified property.

    Cells are colored on a blue-red spectrum based on the relative value
    of the specified property to all other cells. In RANDOM mode, cells
    cycle through a fixed 20-color palette.

    Attributes:
        colorizer (Colorizer): The property by which cells are colored.
        gene (str): Optional, the gene off of which cell color is based.
        range (tuple): Optional, range of values for the colorizer. If provided,
            values are scaled relative to this range instead of min-max normalization.
    """

    def __init__(
        self,
        colorizer: Colorizer = Colorizer.PRESSURE,
        gene: Gene | str = None,
        range: tuple | None = None,
    ):

        self.colorizer = colorizer
        self.gene = gene
        self.range = range
        self.color_map = {}
        self.color_counter = 0

    def _scale(self, values):
        """Scale values using the specified range instead of min-max normalization."""
        if len(values) == 0:
            print("No values to scale")
            return np.array([])

        # Use the specified range (0.5, 2.5) instead of min-max
        min_val, max_val = 0.5, 2.5
        # Clip values to the range
        values = np.clip(values, min_val, max_val)

        # Scale to [0, 1] using the fixed range
        if max_val - min_val == 0:
            print("Warning: max_val - min_val is 0, returning all ones")
            return np.ones_like(values)

        scaled = (values - min_val) / (max_val - min_val)
        return scaled

    def run(self, scene, depsgraph):
        """Applies coloring to cells based on the selected property."""
        cells = self.get_cells()
        if len(cells) == 0:
            return

        red, blue = Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))

        property_values = None
        if self.colorizer != Colorizer.RANDOM:
            property_values = {
                Colorizer.PRESSURE: np.array([
                    cell.pressure if (cell.cloth_mod and
                                    hasattr(cell.cloth_mod, 'settings'))
                    else 0.0 for cell in cells
                ]),
                Colorizer.VOLUME: np.array([cell.volume() for cell in cells]),
                Colorizer.GENE: (np.array([cell.gene_concs[self.gene]
                                         for cell in cells])
                                if self.gene else np.array([])),
            }.get(self.colorizer, None)

            if property_values is not None:
                values = self._scale(property_values)
            else:
                print(f"Error: Invalid colorizer type: {self.colorizer}")
                raise ValueError("Colorizer must be: PRESSURE, VOLUME, GENE, or RANDOM.")
        else:
            # Assign colors in a deterministic sequence from the fixed palette
            for cell in cells:
                if cell.just_divided or cell.name not in self.color_map:
                    self.color_map[cell.name] = COLORS[self.color_counter % len(COLORS)]
                    self.color_counter += 1
            values = [self.color_map[cell.name] for cell in cells]

        # Apply colors to cells
        for cell, value in zip(cells, values, strict=False):
            if self.colorizer == Colorizer.RANDOM:
                color = value
            else:
                color = tuple(blue.lerp(red, value))
            cell.recolor(color)


def _get_divisions(cells: list[Cell]) -> list[tuple[str, str, str]]:
    """Calculate a list of cells that have divided in the past frame.

    Each element of the list contains a tuple of three names: that of the mother
    cell, and then the two daughter cells.

    Args:
        cells: List of cells to check for divisions.

    Returns:
        List of tuples of mother and daughter cell names.
    """
    divisions = set()
    for cell in cells:
        if cell.get("divided"):
            divisions.add(
                (cell.name[:-2], cell.name[:-2] + ".0", cell.name[:-2] + ".1")
            )
    return list(divisions)


@staticmethod
def _contact_area(
    cell1: Cell, cell2: Cell, threshold=0.02
) -> tuple[float, float, float, float]:
    """Calculate the contact areas between two cells.

    Contact is defined as two faces that are within a set threshold distance
    from each other.

    Args:
        cell1: First cell to calculate contact.
        cell2: Second cell to calculate contact.
        threshold: Maximum distance between two faces of either cell to consider
            as contact.

    Returns:
        A tuple containing for elements:
            - Total area of cell1 in contact with cell2
            - Total area of cell2 in contact with cell1
            - Ratio of area of cell1 in contact with cell2
            - Ratio of area of cell2 in contact with cell1
    """
    faces1 = cell1.obj_eval.data.polygons
    faces2 = cell2.obj_eval.data.polygons

    centers1 = [cell1.obj_eval.matrix_world @ f.center for f in faces1]
    centers2 = [cell2.obj_eval.matrix_world @ f.center for f in faces2]

    dists = np.array(cdist(centers1, centers2, "euclidean"))

    contact_faces1 = np.any(dists < threshold, axis=1)
    contact_faces2 = np.any(dists < threshold, axis=0)

    areas1 = np.array([f.area for f in faces1])
    areas2 = np.array([f.area for f in faces2])

    contact_areas1 = np.sum(areas1[contact_faces1])
    contact_areas2 = np.sum(areas2[contact_faces2])

    ratio1 = contact_areas1 / np.sum(areas1)
    ratio2 = contact_areas2 / np.sum(areas2)

    return contact_areas1, contact_areas2, ratio1, ratio2


@staticmethod
def _contact_areas(cells: list[Cell], threshold=4) -> tuple[dict, dict]:
    """Calculate the pairwise contact areas between a list of cells.

    Contact is calculated heuristically by first screening cells that are within
    a certain threshold distance between each other.

    Args:
        cells: The list of cells to calculate contact areas over.
        threshold: The maximum distance between cells to consider them for contact.

    Returns:
        A tuple containing two dictionaries:
            - First dictionary maps cell names to lists of (contact_cell, area) tuples
            - Second dictionary maps cell names to lists of (contact_cell, ratio) tuples
            - For cells with no contacts, a list with a single tuple (None, 0.0) is returned
    """
    coms = [cell.COM() for cell in cells]
    dists = squareform(pdist(coms, "euclidean"))

    mask = dists < threshold
    mask = np.triu(mask, k=1)

    pairs = np.where(mask)

    # Initialize dictionaries with empty lists for all cells
    areas = {cell.name: [] for cell in cells}
    ratios = {cell.name: [] for cell in cells}

    # Update with actual contact values
    for i, j in zip(pairs[0], pairs[1], strict=False):
        contact_area_i, contact_area_j, ratio_i, ratio_j = _contact_area(
            cells[i], cells[j]
        )
        # Only add contacts if they have non-zero area
        if contact_area_i > 0:
            areas[cells[i].name].append((cells[j].name, contact_area_i))
        if contact_area_j > 0:
            areas[cells[j].name].append((cells[i].name, contact_area_j))
        if ratio_i > 0:
            ratios[cells[i].name].append((cells[j].name, ratio_i))
        if ratio_j > 0:
            ratios[cells[j].name].append((cells[i].name, ratio_j))

    # Add zero values for cells with no contacts
    for cell in cells:
        if not areas[cell.name]:
            areas[cell.name] = [(None, 0.0)]
        if not ratios[cell.name]:
            ratios[cell.name] = [(None, 0.0)]

    print(f"Areas: {areas}")
    return areas, ratios


@staticmethod
def _shape_features(cells: list[Cell]) -> tuple[float, float, float, float]:
    """Calculate a set of shape features of a cell.

    Inlcudes the aspect ratio, sphericity

    Args:
        cell: A cell.

    Returns:
        Shape features (aspect ratio, sphericity, compactness, sav_ratio).
    """

    aspect_ratios = []
    sphericities = []
    compactnesses = []
    sav_ratios = []

    for cell in cells:
        if cell.is_collapsed():
            print(f"Deleting collapsed cell: {cell.name}")
            # Delete motion force if it exists
            if hasattr(cell, 'motion_force') and cell.motion_force:
                bpy.data.objects.remove(cell.motion_force.obj, do_unlink=True)
            # Delete the cell
            bpy.data.objects.remove(cell.obj, do_unlink=True)
            continue

        aspect_ratio = cell.aspect_ratio()
        sphericity = cell.sphericity()
        compactness = cell.compactness()
        sav_ratio = cell.sav_ratio()

        aspect_ratios.append(aspect_ratio)
        sphericities.append(sphericity)
        compactnesses.append(compactness)
        sav_ratios.append(sav_ratio)

    return (aspect_ratios, sphericities, compactnesses, sav_ratios)


class _all:
    def __get__(self, instance, cls):
        return ~cls(0)


class DataFlag(Flag):
    """Enum of data flags used by the :func:`DataExporter` handler.

    Attributes:
        TIMES: time elapsed since beginning of simulation.
        DIVISIONS: list of cells that have divided and their daughter cells.
        MOTION_PATH: list of the current position of each cell.
        FORCE_PATH: list of the current positions of the associated
            motion force of each cell.
        VOLUMES: list of the current volumes of each cell.
        PRESSURES: list of the current pressures of each cell.
        CONTACT_AREAS: list of contact areas between each pair of cells.
        CONCENTRATIONS: concentrations of each molecule in the grid system.
    """

    TIMES = auto()
    DIVISIONS = auto()
    MOTION_PATH = auto()
    FORCE_PATH = auto()
    VOLUMES = auto()
    PRESSURES = auto()
    CONTACT_AREAS = auto()
    SHAPE_FEATURES = auto()
    GRID = auto()
    CELL_CONCENTRATIONS = auto()

    ALL = _all()


class DataExporter(Handler):
    """Handler for the reporting and saving of data generated during the simulation."""

    def __init__(self, path="", options: DataFlag = DataFlag.ALL):
        self.path = path
        self.options = options
        self.h5file = None

    @override
    def setup(
        self,
        get_cells: Callable[[], list[Cell]],
        get_diffsystems: Callable[[], list[DiffusionSystem]],
        dt
    ) -> None:
        super().setup(get_cells, get_diffsystems, dt)
        self.time_start = datetime.now()

        if self.path:
            if os.path.exists(self.path):
                try:
                    os.remove(self.path)
                except Exception as e:
                    print(f"Could not remove existing file: {e}")
                    raise

            self.h5file = h5py.File(self.path, 'w')
            self.h5file.attrs['seed'] = bpy.context.scene["seed"]
            self.frames_group = self.h5file.create_group('frames')
        else:
            pass

    @override
    def run(self, scene, depsgraph) -> None:
        frame_number = scene.frame_current
        frame_group_name = f'frame_{frame_number:03d}'

        if self.path:
            # Check if the group already exists
            if frame_group_name in self.frames_group:
                print(f"Group {frame_group_name} already exists. Delelting, recreating")
                del self.frames_group[frame_group_name]  # Remove the existing group
            frame_group = self.frames_group.create_group(frame_group_name)
            frame_group.attrs['frame_number'] = frame_number
        else:
            frame_out = {"frame": frame_number}

        if self.options & DataFlag.TIMES:
            time_elapsed = (datetime.now() - self.time_start).total_seconds()
            if self.path:
                frame_group.attrs['time'] = time_elapsed
            else:
                frame_out["time"] = time_elapsed

        if self.options & DataFlag.DIVISIONS:
            divisions = _get_divisions(self.get_cells())
            if self.path:
                frame_group.create_dataset('divisions', data=divisions)
            else:
                frame_out["divisions"] = divisions

        # Collect cell data
        cells = self.get_cells()
        if self.path:
            cells_group = frame_group.create_group('cells')
        else:
            frame_out["cells"] = []

        for cell in cells:
            cell_name = cell.name
            if self.path:
                cell_group = cells_group.create_group(cell_name)
            else:
                cell_out = {"name": cell_name}

            if self.options & DataFlag.MOTION_PATH:
                loc = np.array(cell.loc, dtype=np.float64)  # Ensure loc is a NumPy array
                if self.path:
                    cell_group.create_dataset('loc', data=loc)
                else:
                    cell_out["loc"] = loc.tolist()

            if self.options & DataFlag.FORCE_PATH:
                motion_loc = np.array(cell.motion_force.loc, dtype=np.float64)
                if self.path:
                    cell_group.create_dataset('motion_loc', data=motion_loc)
                else:
                    cell_out["motion_loc"] = motion_loc.tolist()

            if self.options & DataFlag.VOLUMES:
                volume = float(cell.volume())  # Convert volume to a float
                if self.path:
                    cell_group.attrs['volume'] = volume
                else:
                    cell_out["volume"] = volume

            if self.options & DataFlag.PRESSURES and cell.physics_enabled:
                pressure = float(cell.pressure)  # Ensure pressure is a float
                if self.path:
                    cell_group.attrs['pressure'] = pressure
                else:
                    cell_out["pressure"] = pressure

            if self.options & DataFlag.CELL_CONCENTRATIONS:
                try:
                    if isinstance(cell.molecules_conc, dict):
                        # Convert dictionary values to an array
                        concentrations = np.array(list(cell.molecules_conc.values()),
                                                  dtype=np.float64
                                                  )
                    else:
                        concentrations = np.array(cell.molecules_conc, dtype=np.float64)

                    if self.path:
                        cell_group.create_dataset('concentrations', data=concentrations)
                    else:
                        cell_out["concentrations"] = concentrations.tolist()
                except Exception as e:
                    print(f"Error saving concentrations for cell {cell_name}: {e}")

            if not self.path:
                frame_out["cells"].append(cell_out)

        if self.options & DataFlag.SHAPE_FEATURES:
            aspect_ratios, sphericities, \
                compactnesses, sav_ratios = _shape_features(cells)
            if self.path:
                frame_group.create_dataset(
                    'aspect_ratios',
                    data=np.array(aspect_ratios, dtype=np.float64)
                )
                frame_group.create_dataset(
                    'sphericities',
                    data=np.array(sphericities, dtype=np.float64)
                )
                frame_group.create_dataset(
                    'compactnesses',
                    data=np.array(compactnesses, dtype=np.float64)
                )
                frame_group.create_dataset(
                    'sav_ratios',
                    data=np.array(sav_ratios, dtype=np.float64)
                )
            else:
                frame_out["aspect_ratios"] = aspect_ratios.tolist()
                frame_out["sphericities"] = sphericities.tolist()
                frame_out["compactnesses"] = compactnesses.tolist()
                frame_out["sav_ratios"] = sav_ratios.tolist()

        # Handle contact areas
        if self.options & DataFlag.CONTACT_AREAS:
            try:
                areas, ratios = _contact_areas(cells)

                # Create a mapping of cell names to indices for efficient storage
                cell_names = {cell.name: idx for idx, cell in enumerate(cells)}
                n_cells = len(cell_names)

                # Initialize arrays for areas and ratios with zeros
                # Create a full matrix of zeros for all possible cell pairs
                areas_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
                ratios_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)

                # Process areas if they exist
                if areas and isinstance(areas, dict):
                    for cell_name, contacts in areas.items():
                        if contacts and cell_name in cell_names:
                            cell1_idx = cell_names[cell_name]
                            for contact_cell, area in contacts:
                                if (isinstance(area, int | float) and
                                    contact_cell in cell_names):
                                    cell2_idx = cell_names[contact_cell]
                                    areas_matrix[cell1_idx, cell2_idx] = float(area)
                                    areas_matrix[cell2_idx, cell1_idx] = float(area)  # Symmetric

                # Process ratios if they exist
                if ratios and isinstance(ratios, dict):
                    for cell_name, contacts in ratios.items():
                        if contacts and cell_name in cell_names:
                            cell1_idx = cell_names[cell_name]
                            for contact_cell, ratio in contacts:
                                if (isinstance(ratio, int | float) and
                                    contact_cell in cell_names):
                                    cell2_idx = cell_names[contact_cell]
                                    ratios_matrix[cell1_idx, cell2_idx] = float(ratio)
                                    ratios_matrix[cell2_idx, cell1_idx] = float(ratio)  # Symmetric

                # Convert to structured arrays with minimal memory footprint
                # Only store non-zero values to save space
                areas_data = []
                ratios_data = []

                # Get indices of non-zero elements
                areas_nonzero = np.nonzero(areas_matrix)
                ratios_nonzero = np.nonzero(ratios_matrix)

                # Store non-zero values
                for i, j in zip(*areas_nonzero, strict=False):
                    if i < j:  # Only store upper triangle to avoid duplicates
                        areas_data.append((i, j, areas_matrix[i, j]))

                for i, j in zip(*ratios_nonzero, strict=False):
                    if i < j:  # Only store upper triangle to avoid duplicates
                        ratios_data.append((i, j, ratios_matrix[i, j]))

                # Convert to structured arrays
                if areas_data:
                    areas = np.array(areas_data, dtype=[
                        ('cell1_idx', 'i4'),  # 32-bit integer
                        ('cell2_idx', 'i4'),
                        ('area', 'f4')        # 32-bit float
                    ])
                else:
                    areas = np.array([], dtype=[
                        ('cell1_idx', 'i4'),
                        ('cell2_idx', 'i4'),
                        ('area', 'f4')
                    ])

                if ratios_data:
                    ratios = np.array(ratios_data, dtype=[
                        ('cell1_idx', 'i4'),
                        ('cell2_idx', 'i4'),
                        ('ratio', 'f4')
                    ])
                else:
                    ratios = np.array([], dtype=[
                        ('cell1_idx', 'i4'),
                        ('cell2_idx', 'i4'),
                        ('ratio', 'f4')
                    ])

                if self.path:
                    # Save cell names mapping
                    frame_group.create_dataset('cell_names',
                                             data=np.array(list(cell_names.keys()),
                                                         dtype='S50'))

                    # Save contact data
                    frame_group.create_dataset('contact_areas', data=areas)
                    frame_group.create_dataset('contact_ratios', data=ratios)

                    # Save the full matrices for completeness
                    frame_group.create_dataset('contact_areas_matrix', data=areas_matrix)
                    frame_group.create_dataset('contact_ratios_matrix', data=ratios_matrix)
                else:
                    # For non-file output, convert back to cell names
                    areas_list = []
                    for cell1_idx, cell2_idx, area in areas:
                        areas_list.append({
                            'cell1': list(cell_names.keys())[cell1_idx],
                            'cell2': list(cell_names.keys())[cell2_idx],
                            'area': float(area)
                        })

                    ratios_list = []
                    for cell1_idx, cell2_idx, ratio in ratios:
                        ratios_list.append({
                            'cell1': list(cell_names.keys())[cell1_idx],
                            'cell2': list(cell_names.keys())[cell2_idx],
                            'ratio': float(ratio)
                        })

                    frame_out["contact_areas"] = areas_list
                    frame_out["contact_ratios"] = ratios_list
            except Exception as e:
                print(f"Error saving contact areas for frame {frame_number}: {e}")

        # Handle GRID data
        if self.options & DataFlag.GRID:
            for diff_system in self.get_diff_systems():
                try:
                    # Ensure grid concentrations are converted to NumPy arrays
                    grid_conc = np.array(diff_system._grid_concentrations,
                                         dtype=np.float64
                                         )
                    for mol in diff_system._molecules:
                        mol_name = mol._name
                        if self.path:
                            mol_group = frame_group.require_group(mol_name)
                            mol_group.create_dataset('concentrations', data=grid_conc)
                        else:
                            if mol_name not in frame_out:
                                frame_out[mol_name] = {"concentrations": grid_conc.tolist()}
                except Exception as e:
                    print(f"Error saving grid concentrations: {e}")

        if not self.path:
            print(frame_out)

    def close(self):
        """Close the HDF5 file."""
        if self.h5file:
            self.h5file.close()
            self.h5file = None

    def __del__(self):
        """Ensure HDF5 file is closed when object is deleted."""
        self.close()


class SliceExporter(Handler):
    """Handler to save point cloud data of the simulation at each frame and convert to 3D array.

    Args:
        output_dir: The directory to save the point cloud data.
        resolution: The resolution of the grid to save the point cloud data.
            Default is (256, 256, 256).
        scale: The scale of the grid to save the point cloud data.
            Default is (0.2, 0.2, 0.2).
        microscope_dt: The time step of the microscope.
            Default is 10.
        padding: The padding of the grid to save the point cloud data.
            Default is 5.0.
        downscale: The downscaling factor for the low-resolution version.
            Can be a single integer or a tuple of integers for each dimension.
            Default is None (no downscaling).
    """

    def __init__(
        self,
        output_dir: str = "",
        resolution: tuple[int, int, int] = (512, 512, 512),
        scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
        microscope_dt: int = 10,
        padding: float = 5.0,
        downscale: int | tuple[int, int, int] | None = None
    ):
        self.output_dir = output_dir
        self.resolution = resolution
        self.scale = scale
        self.microscope_dt = microscope_dt
        self.padding = padding
        self.downscale = downscale

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of all visible mesh objects in the scene.

        Returns:
            tuple: (min_coords, max_coords) representing the bounding box
        """
        visible_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'
                           and obj.visible_get()]

        if not visible_objects:
            return np.zeros(3), np.zeros(3)

        # Get world space coordinates of all vertices
        all_vertices = []
        for obj in visible_objects:
            mesh = obj.data
            world_matrix = obj.matrix_world
            for v in mesh.vertices:
                world_co = world_matrix @ v.co
                all_vertices.append(world_co)

        vertices_array = np.array(all_vertices)
        min_coords = np.min(vertices_array, axis=0)
        max_coords = np.max(vertices_array, axis=0)

        # Add padding
        padding_array = np.array([self.padding] * 3)
        min_coords -= padding_array
        max_coords += padding_array

        return min_coords, max_coords

    def world_to_grid_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid coordinates.

        Args:
            world_coords: World space coordinates

        Returns:
            Grid coordinates
        """
        min_coords, max_coords = self.get_scene_bounds()
        # Normalize coordinates to [0, 1] range
        normalized = (world_coords - min_coords) / (max_coords - min_coords)
        # Scale to grid size
        return normalized * (np.array(self.resolution) - 1)

    def sample_mesh_points(self, obj: bpy.types.Object, num_points: int = 200000) -> np.ndarray:
        """Sample points densely and uniformly from a mesh object.

        Args:
            obj: The mesh object to sample from
            num_points: Number of points to sample (increased for better coverage)

        Returns:
            Array of sampled points in world coordinates
        """
        # Get the mesh data
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        # Calculate total area for weighted sampling
        total_area = sum(f.calc_area() for f in bm.faces)

        # Calculate points per face based on area
        points_per_face = []
        for face in bm.faces:
            face_area = face.calc_area()
            # Ensure at least 10 points per face for small faces
            num_face_points = max(20, int(num_points * (face_area / total_area)))
            points_per_face.append(num_face_points)

        # Sample points
        points = []
        for face, num_face_points in zip(bm.faces, points_per_face, strict=False):
            # Get face vertices
            verts = face.verts
            if len(verts) > 3:
                # For non-triangular faces, use fan triangulation
                center = Vector(face.calc_center_median())
                for i in range(len(verts)):
                    v1 = Vector(verts[i].co)
                    v2 = Vector(verts[(i + 1) % len(verts)].co)
                    v3 = center

                    # Sample points in this triangle
                    for _ in range(num_face_points // len(verts)):
                        # Generate random barycentric coordinates
                        r1, r2 = float(np.random.random()), float(np.random.random())
                        if r1 + r2 > 1:
                            r1, r2 = 1 - r1, 1 - r2

                        # Calculate point position
                        point = v1 + v2 * r1 - v1 * r1 + v3 * r2 - v1 * r2
                        points.append(obj.matrix_world @ point)
            else:
                # For triangular faces
                v1 = Vector(verts[0].co)
                v2 = Vector(verts[1].co)
                v3 = Vector(verts[2].co)

                # Sample points in this triangle
                for _ in range(num_face_points):
                    # Generate random barycentric coordinates
                    r1, r2 = float(np.random.random()), float(np.random.random())
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2

                    # Calculate point position
                    point = v1 + v2 * r1 - v1 * r1 + v3 * r2 - v1 * r2
                    points.append(obj.matrix_world @ point)

        bm.free()
        return np.array(points)

    def points_to_volume(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to a 3D volume array with continuous surfaces.

        Args:
            points: Array of points in world coordinates

        Returns:
            3D volume array with continuous surfaces
        """
        # Convert points to grid coordinates
        grid_points = self.world_to_grid_coords(points)
        volume = np.zeros(self.resolution, dtype=np.float32)
        grid_points = grid_points.astype(int)
        mask = np.all((grid_points >= 0) & (grid_points < self.resolution), axis=1)
        grid_points = grid_points[mask]
        volume[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]] = 1.0
        volume = ndimage.binary_dilation(volume, structure=np.ones((2,2,2)))

        return volume

    def downsample_volume(self, volume: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Downsample the volume array and adjust the scale accordingly.

        Args:
            volume: The original volume array

        Returns:
            tuple: (downsampled_volume, new_scale)
        """
        if self.downscale is None:
            return volume, self.scale

        if isinstance(self.downscale, int):
            downscale = (self.downscale, self.downscale, self.downscale)
        else:
            downscale = self.downscale

        # Calculate new scale
        new_scale = tuple(s * d for s, d in zip(self.scale, downscale, strict=False))

        # Downsample using sum operation (preserves binary nature)
        from scipy import ndimage
        downsampled = ndimage.zoom(volume,
                                 (1/downscale[0], 1/downscale[1], 1/downscale[2]),
                                 order=0)  # order=0 for nearest neighbor

        return downsampled, new_scale

    def run(self, scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph):
        """Run the point cloud export process for each frame.

        Args:
            scene: The Blender scene.
            depsgraph: The dependency graph.
        """
        # export every microscope_dt
        if scene.frame_current % self.microscope_dt != 0:
            return

        # Determine the time step index based on the current frame
        time_step = scene.frame_current // self.microscope_dt

        # Create a folder for this time step
        time_step_dir = os.path.join(self.output_dir, f"T{time_step:03d}")
        if not os.path.exists(time_step_dir):
            os.makedirs(time_step_dir)

        # Get all visible mesh objects
        visible_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'
                           and obj.visible_get()]

        # Sample points from all meshes
        all_points = []
        for obj in visible_objects:
            points = self.sample_mesh_points(obj)
            all_points.append(points)

        # Combine all points
        combined_points = np.vstack(all_points)

        # Convert to volume
        volume = self.points_to_volume(combined_points)

        # Create coordinate arrays based on scale
        z_coords = np.arange(self.resolution[0]) * self.scale[0]
        y_coords = np.arange(self.resolution[1]) * self.scale[1]
        x_coords = np.arange(self.resolution[2]) * self.scale[2]

        # Create dataset
        ds = xr.Dataset(
            data_vars={
                'volume': (['z', 'y', 'x'], volume)
            },
            coords={
                'z': z_coords,
                'y': y_coords,
                'x': x_coords
            },
            attrs={
                'description': '3D volume representation of point cloud data',
                'units': 'microns',
                'frame': scene.frame_current,
                'num_points': len(combined_points)
            }
        )

        # Save full resolution to NetCDF
        output_path = os.path.join(time_step_dir, "point_cloud_volume.nc")
        ds.to_netcdf(output_path)

        # Save downsampled version if requested
        if self.downscale is not None:
            downsampled_volume, new_scale = self.downsample_volume(volume)
            z_coords_ds = np.arange(downsampled_volume.shape[0]) * new_scale[0]
            y_coords_ds = np.arange(downsampled_volume.shape[1]) * new_scale[1]
            x_coords_ds = np.arange(downsampled_volume.shape[2]) * new_scale[2]
            # Create dataset for downsampled version
            ds_ds = xr.Dataset(
                data_vars={
                    'volume': (['z', 'y', 'x'], downsampled_volume)
                },
                coords={
                    'z': z_coords_ds,
                    'y': y_coords_ds,
                    'x': x_coords_ds
                },
                attrs={
                    'description': 'Downsampled 3D volume representation of point cloud data',
                    'units': 'microns',
                    'frame': scene.frame_current,
                    'num_points': len(combined_points),
                    'downscale_factor': self.downscale
                }
            )

            # Save downsampled version to NetCDF
            output_path_ds = os.path.join(time_step_dir, "point_cloud_volume_downsampled.nc")
            ds_ds.to_netcdf(output_path_ds)

        print(f"Saved point cloud volumes for frame {scene.frame_current}")
