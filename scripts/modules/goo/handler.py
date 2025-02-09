from typing import Callable, Union
from typing_extensions import override, Optional
from abc import ABC, abstractmethod

from enum import Enum, Flag, auto
from datetime import datetime
import json
import os
import h5py

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import laplace

import bpy
import bmesh
from mathutils import Vector
from goo.cell import Cell
from goo.gene import Gene
from goo.molecule import Molecule, DiffusionSystem


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
    """Handler for stopping the simulation at the end of the simulation time."""

    def run(self, scene, depsgraph):
        # Check if the current frame is the last frame
        if scene.frame_current >= bpy.context.scene.frame_end:
            print(f"Simulation has reached the last frame: {scene.frame_current}. \
                  Stopping.")
            # bpy.app.handlers.frame_change_post.remove(self.run)
            bpy.ops.screen.animation_cancel(restore_frame=True) 
            for cell in self.get_cells():
                # freeze tissue at end of simulation
                cell.disable_physics()
                cell.remesh()
        else:
            frame_str = f"Calculating frame {scene.frame_current}"
            total_length = len(frame_str) + 8
            border_line = "=" * total_length

            print(border_line)
            print(f"=== {frame_str} ===")
            print(border_line)


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
            if not cell.physics_enabled:
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
        super(DiffusionHandler, self).setup(get_cells, get_diffsystems, dt)
        self.get_diffsystem().build_kdtree()

    def run(self, scene, depsgraph) -> None:
        self.get_diffsystem().simulate_diffusion()


class NetworkHandler(Handler):
    """Handler for gene regulatory networks."""

    def run(self, scene, despgraph):
        for cell in self.get_cells():
            cell.step_grn(self.get_diffsystem())


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

            cell_size = cell.major_axis().length() / 2
            for force in cell.adhesion_forces:
                if not force.enabled():
                    continue
                force.min_dist = cell_size - 0.4
                force.max_dist = cell_size + 0.4

            if cell.motion_force:
                cell.move()
                
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
        range (tuple): Optional, range of values for the colorizer.
    """

    def __init__(
        self,
        colorizer: Colorizer = Colorizer.PRESSURE,
        gene: Union[Gene, str] = None,
        range: Optional[tuple] = None,
    ):
        self.colorizer = colorizer
        self.gene = gene
        self.range = range
        self.color_map = {}
        self.color_counter = 0

    def _scale(self, values):
        if len(values) == 0:
            return np.array([])
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val == 0:
            return np.ones_like(values)
        return (values - min_val) / (max_val - min_val)

    def run(self, scene, depsgraph):
        """Applies coloring to cells based on the selected property."""
        cells = self.get_cells()
        red, blue = Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))

        property_values = {
            Colorizer.PRESSURE: np.array([cell.pressure for cell in cells]),
            Colorizer.VOLUME: np.array([cell.volume() for cell in cells]),
            Colorizer.GENE: np.array([cell.metabolites[self.gene] for cell in cells]) 
            if self.gene else np.array([]),
        }.get(self.colorizer, None)

        if self.colorizer == Colorizer.RANDOM:
            # Assign colors in a deterministic sequence from the fixed palette
            for cell in cells:
                if cell.just_divided or cell.name not in self.color_map:
                    self.color_map[cell.name] = COLORS[self.color_counter % len(COLORS)]
                    self.color_counter += 1  # Move to the next color
            values = [self.color_map[cell.name] for cell in cells]
        elif property_values is not None:
            values = self._scale(property_values)
        else:
            raise ValueError("Colorizer must be: PRESSURE, VOLUME, GENE, or RANDOM.")

        # Apply colors to cells
        for cell, value in zip(cells, values):
            cell.recolor(value if self.colorizer == Colorizer.RANDOM 
                         else tuple(blue.lerp(red, value)))
                

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
        if "divided" in cell and cell["divided"]:
            divisions.add(
                (cell.name[:-2], cell.name[:-2] + ".0", cell.name[:-2] + ".1")
            )
    return list(divisions)


@staticmethod
def _contact_area(
    cell1: Cell, cell2: Cell, threshold=0.1
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
        A list of tuples containing pairwise contact areas and contact ratios.
            See :func:`_contact_area`.
    """
    coms = [cell.COM() for cell in cells]
    dists = squareform(pdist(coms, "euclidean"))

    mask = dists < threshold
    mask = np.triu(mask, k=1)

    pairs = np.where(mask)

    areas = {cell.name: [] for cell in cells}
    ratios = {cell.name: [] for cell in cells}
    for i, j in zip(pairs[0], pairs[1]):
        contact_area_i, contact_area_j, ratio_i, ratio_j = _contact_area(
            cells[i], cells[j]
        )
        areas[cells[i].name].append((cells[j].name, contact_area_i))
        areas[cells[j].name].append((cells[i].name, contact_area_j))
        ratios[cells[i].name].append((cells[j].name, ratio_i))
        ratios[cells[j].name].append((cells[i].name, ratio_j))

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
        super(DataExporter, self).setup(get_cells, get_diffsystems, dt)
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
            print({"seed": bpy.context.scene["seed"], "frames": []})

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
                # If areas or ratios are dictionaries, extract numerical values
                if isinstance(areas, dict):
                    areas = np.array(list(areas.values()), dtype=np.float64)
                else:
                    areas = np.array(areas, dtype=np.float64)

                if isinstance(ratios, dict):
                    ratios = np.array(list(ratios.values()), dtype=np.float64)
                else:
                    ratios = np.array(ratios, dtype=np.float64)

                if self.path:
                    frame_group.create_dataset('contact_areas', data=areas)
                    frame_group.create_dataset('contact_ratios', data=ratios)
                else:
                    frame_out["contact_areas"] = areas.tolist()
                    frame_out["contact_ratios"] = ratios.tolist()
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
            print("Closing HDF5 file.")
            self.h5file.close()
            self.h5file = None

    def __del__(self):
        """Ensure HDF5 file is closed when object is deleted."""
        self.close()