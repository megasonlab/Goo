import os

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum, Flag, auto

import bmesh
import bpy
import h5py
import numpy as np
import xarray as xr

from mathutils import Vector
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist, squareform
from typing_extensions import override

from goo.cell import Cell
from goo.force import create_boundary
from goo.gene import Gene
from goo.molecule import DiffusionSystem, Molecule


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


class ConcentrationVisualizationHandler(Handler):
    """Visualizes a 3D concentration array using color-coded transparent instanced cubes in Blender."""

    def __init__(self, spacing: float | None = None):
        super().__init__()
        self.get_cells = None
        self.get_diffsystem = None
        self.dt = None
        self.spacing = spacing
        self.name = "ConcentrationVisualization"
        self.instance_scale = 1

    def setup(
        self,
        get_cells: Callable[[], list],
        get_diffsystem: Callable[[], object],
        dt: float,
    ) -> None:
        self.get_cells = get_cells
        self.get_diffsystem = get_diffsystem
        self.dt = dt

    def run(self, scene, depsgraph):
        if not self.get_diffsystem:
            print("Warning: ConcentrationVisualizationHandler not initialized. Call setup() first.")
            return

        grid_conc_dict = self.get_diffsystem()._grid_concentrations
        first_mol = next(iter(grid_conc_dict.values()))
        grid_shape = first_mol.shape

        grid_conc = np.zeros(grid_shape, dtype=np.float64)
        for conc in grid_conc_dict.values():
            grid_conc += np.array(conc, dtype=np.float64)

        self._remove_existing_object()
        obj = self._create_pointcloud(grid_conc)
        self._add_geometry_nodes(obj)

    def _remove_existing_object(self):
        existing = bpy.data.objects.get(self.name)
        if existing:
            bpy.data.objects.remove(existing, do_unlink=True)

    def _create_pointcloud(self, grid_conc):
        grid_conc = grid_conc[::2, ::2, ::2]
        nx, ny, nz = grid_conc.shape
        diff_system = self.get_diffsystem()

        xlim, ylim, zlim = (np.array(diff_system.grid_size) - 1) / 2 * diff_system.element_size
        grid_center = diff_system.grid_center

        x = np.linspace(-xlim, xlim, nx) + grid_center[0]
        y = np.linspace(-ylim, ylim, ny) + grid_center[1]
        z = np.linspace(-zlim, zlim, nz) + grid_center[2]

        values = grid_conc.flatten()
        values_norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-9)

        verts = [(x[i], y[j], z[k]) for i in range(nx) for j in range(ny) for k in range(nz)]

        mesh = bpy.data.meshes.new(self.name + "_mesh")
        mesh.from_pydata(verts, [], [])
        mesh.update()

        conc_layer = mesh.attributes.new(name="conc", type='FLOAT', domain='POINT')
        conc_layer.data.foreach_set("value", values_norm)

        obj = bpy.data.objects.new(self.name, mesh)
        bpy.context.collection.objects.link(obj)

        return obj

    def _add_geometry_nodes(self, obj):
        geo_node = obj.modifiers.new("GeoNodes", type='NODES')
        nt = geo_node.node_group = bpy.data.node_groups.new("GN_VisualizeConc", 'GeometryNodeTree')

        nodes, links = nt.nodes, nt.links
        nodes.clear()

        input_node = nodes.new("NodeGroupInput")
        output_node = nodes.new("NodeGroupOutput")
        nt.interface.new_socket("Geometry", socket_type='NodeSocketGeometry', in_out='INPUT')
        nt.interface.new_socket("Geometry", socket_type='NodeSocketGeometry', in_out='OUTPUT')

        attr_node = nodes.new("GeometryNodeInputNamedAttribute")
        attr_node.data_type = 'FLOAT'
        attr_node.inputs["Name"].default_value = "conc"

        color_ramp = nodes.new("ShaderNodeValToRGB")
        color_ramp.color_ramp.elements[0].color = (0, 0, 1, 0.02)
        color_ramp.color_ramp.elements[1].color = (1, 0, 0, 0.02)

        store_color_node = nodes.new("GeometryNodeStoreNamedAttribute")
        store_color_node.data_type = 'FLOAT_COLOR'
        store_color_node.domain = 'POINT'
        store_color_node.inputs["Name"].default_value = "color"

        instance_node = nodes.new("GeometryNodeInstanceOnPoints")
        cube_obj = self._get_or_create_cube()
        obj_info = nodes.new("GeometryNodeObjectInfo")
        obj_info.inputs["Object"].default_value = cube_obj

        realize_node = nodes.new("GeometryNodeRealizeInstances")

        set_material_node = nodes.new("GeometryNodeSetMaterial")
        set_material_node.inputs["Material"].default_value = self._get_or_create_material()

        links.new(input_node.outputs["Geometry"], store_color_node.inputs["Geometry"])
        links.new(attr_node.outputs["Attribute"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], store_color_node.inputs["Value"])  # FIXED connection here
        links.new(store_color_node.outputs["Geometry"], instance_node.inputs["Points"])
        links.new(obj_info.outputs["Geometry"], instance_node.inputs["Instance"])
        links.new(instance_node.outputs["Instances"], realize_node.inputs["Geometry"])
        links.new(realize_node.outputs["Geometry"], set_material_node.inputs["Geometry"])
        links.new(set_material_node.outputs["Geometry"], output_node.inputs["Geometry"])

        input_node.location = (-600, 0)
        attr_node.location = (-400, 200)
        color_ramp.location = (-200, 200)
        store_color_node.location = (0, 100)
        obj_info.location = (-400, -200)
        instance_node.location = (200, 0)
        realize_node.location = (400, 0)
        set_material_node.location = (600, 0)
        output_node.location = (800, 0)


    def _get_or_create_material(self):
        mat_name = "ConcMaterial"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            mat.blend_method = 'BLEND'

            # Handle shadow method for different Blender versions
            if hasattr(mat, 'shadow_method'):
                mat.shadow_method = 'HASHED'
            else:
                # For Blender 4.5+, use the new shadow method
                mat.shadow_method = 'CLIP'  # or 'NONE' depending on your needs

            # Vertex color attribute
            attribute_node = mat.node_tree.nodes.new("ShaderNodeAttribute")
            attribute_node.attribute_name = "color"

            # Connect attribute color to BSDF
            mat.node_tree.links.new(attribute_node.outputs["Color"], bsdf.inputs["Base Color"])
            mat.node_tree.links.new(attribute_node.outputs["Alpha"], bsdf.inputs["Alpha"])
        return mat


    def _get_or_create_cube(self):
        obj = bpy.data.objects.get("CubeInstance")
        if obj is None:
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.active_object
            obj.name = "CubeInstance"
            obj.hide_render = True
            obj.hide_viewport = True

            # Ensure object is selected and active before changing mode
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            # Optimize cube geometry
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.delete(type='ONLY_FACE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.edge_face_add()
            bpy.ops.object.mode_set(mode='OBJECT')

        return obj



class StopHandler(Handler):
    """Handler for stopping the simulation at the end of the simulation time or when reaching max cells."""

    def __init__(self, max_cells=None):
        self.max_cells = max_cells

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

            try:
                # Store the current context
                current_context = bpy.context.area
                current_mode = bpy.context.mode if hasattr(bpy.context, 'mode') else None

                # Freeze all cells
                for cell in self.get_cells():
                    # Apply all modifiers to get the final state
                    for mod in cell.obj.modifiers:
                        try:
                            # Ensure we're in object mode and the object is active
                            if bpy.context.mode != 'OBJECT':
                                bpy.ops.object.mode_set(mode='OBJECT')
                            bpy.context.view_layer.objects.active = cell.obj
                            cell.obj.select_set(True)
                            bpy.ops.object.modifier_apply(modifier=mod.name)
                        except Exception as e:
                            print(f"Warning: Could not apply modifier {mod.name} to {cell.name}: {e}")
                    cell.disable_physics()
                    cell.remesh()
            except Exception as e:
                print(f"Warning: Could not freeze cells properly: {e}")
                # Still try to disable physics on all cells
                for cell in self.get_cells():
                    if cell.physics_enabled:
                        try:
                            cell.disable_physics()
                        except Exception:
                            pass

            # Remove all handlers
            for handler in bpy.app.handlers.frame_change_pre[:]:
                bpy.app.handlers.frame_change_pre.remove(handler)
            for handler in bpy.app.handlers.frame_change_post[:]:
                bpy.app.handlers.frame_change_post.remove(handler)

            # Useful when not using sim.run()
            bpy.context.scene.frame_set(1)
            try:
                # Try to cancel animation only if we're in a valid context
                if bpy.context.area and bpy.context.area.type == 'VIEW_3D':
                    bpy.ops.screen.animation_cancel()
            except Exception as e:
                print(f"Warning: Could not cancel animation: {e}")


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


class MolecularHandler(Handler):
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
        for mol in self.get_diffsystem().molecules:
            self.get_diffsystem().simulate_diffusion(mol)

            for cell in self.get_cells():
                # cell.sense(mol=mol)
                # cell.secrete(mol=mol)
                cell.update_physics_with_molecules(mol=mol)

class NetworkHandler(Handler):
    """Handler for gene regulatory networks."""

    def run(self, scene, despgraph):
        for cell in self.get_cells():
            cell.step_grn(dt=self.dt)
            cell.update_physics_with_grn()

class BoundaryHandler(Handler):
    """Handler for updating boundary volume over time."""

    def __init__(
        self,
        loc: tuple[float, float, float] = (0, 0, 0),
        radius: float = 10,
    ):
        self.loc = loc
        self.radius = radius
        self.boundary = create_boundary(loc, radius)

    def run(self, scene, depsgraph):
        volume = self.boundary.update_volume()
        self.boundary.remesh(voxel_size=1)
        bpy.context.scene.world["boundary_volume"] = volume
        print(f"Boundary volume: {volume}")

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
        sphericities = []
        for cell in cells:
            sphericity = cell.sphericity()
            if sphericity is not None:
                sphericities.append(sphericity)
        average_sphericity = np.mean(sphericities) if sphericities else 0

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
                    force.loc = cell.loc

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
Colorizer = Enum("Colorizer", ["PRESSURE", "VOLUME", "RANDOM", "GENE", "MOLECULE", "LINEAGE", "LINEAGE_DISTANCE"])

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
        metabolite: Gene | Molecule | str = None,
        range: tuple | None = None,
    ):
        self.colorizer = colorizer
        self.metabolite = metabolite
        self.range = range
        self.color_map = {}
        self.color_counter = 0
        self.lineage_colors = {}  # Store lineage-based colors
        # Inferno-like colormap parameters
        self.base_hue = 0.8  # Start with purple (0.8)
        self.hue_step = 0.15  # Larger step for more distinct colors
        self.base_saturation = 0.9  # High saturation for vibrant colors
        self.saturation_step = 0.05  # Smaller step to maintain vibrancy
        self.value_step = 0.1  # Step for value changes

    def _get_lineage_path(self, cell_name: str) -> list[int]:
        """Get the lineage path as a list of 0s and 1s from root to cell."""
        if len(cell_name) <= 2:  # Root cell
            return []
        # Extract the path from the cell name
        # Example: "cell.0.1.0" -> [0, 1, 0]
        path = []
        parts = cell_name.split('.')
        for part in parts[1:]:  # Skip the first part (cell name)
            path.append(int(part))
        return path

    def _path_to_color(self, path: list[int]) -> tuple[float, float, float]:
        """Convert a lineage path to an inferno-like color using HSV."""
        if not path:
            return (self.base_hue, self.base_saturation, 0.3)  # Dark purple for root
        # base color
        hue = self.base_hue
        for i, step in enumerate(path):
            # Contribution decreases with depth
            contribution = self.hue_step / (2 ** i)
            if step == 0:
                hue = (hue - contribution) % 1.0  # towards red
            else:
                hue = (hue + contribution) % 1.0  # towards yellow
        # saturation and value
        depth = len(path)
        saturation = min(1.0, self.base_saturation + depth * self.saturation_step)
        value = min(1.0, 0.3 + depth * self.value_step)  # starts dark, get brighter

        return (hue, saturation, value)

    def _hsv_to_rgb(self, hsv):
        """Convert HSV color to RGB."""
        h, s, v = hsv
        if s == 0.0:
            return (v, v, v)

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        if i == 0:
            return (v, t, p)
        elif i == 1:
            return (q, v, p)
        elif i == 2:
            return (p, v, t)
        elif i == 3:
            return (p, q, v)
        elif i == 4:
            return (t, p, v)
        else:
            return (v, p, q)

    def _assign_lineage_color(self, cell):
        """Assign a color based on lineage path using inferno-like colormap."""
        if cell.name in self.lineage_colors:
            return self.lineage_colors[cell.name]

        cell_path = self._get_lineage_path(cell.name)
        color = self._path_to_color(cell_path)
        self.lineage_colors[cell.name] = color
        return color

    def run(self, scene, depsgraph):
        """Applies coloring to cells based on the selected property."""
        cells = self.get_cells()
        if len(cells) == 0:
            return

        red, blue = Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))

        property_values = None
        if self.colorizer != Colorizer.RANDOM:
            if self.colorizer == Colorizer.PRESSURE:
                property_values = np.array([
                    cell.pressure if (cell.cloth_mod and
                                    hasattr(cell.cloth_mod, 'settings'))
                    else 0.0 for cell in cells
                ])
            elif self.colorizer == Colorizer.VOLUME:
                property_values = np.array([cell.volume() for cell in cells])
            elif self.colorizer == Colorizer.GENE:
                property_values = (np.array([cell.gene_concs[self.metabolite]
                                         for cell in cells])
                                if self.metabolite else np.array([]))
            elif self.colorizer == Colorizer.MOLECULE:
                property_values = (np.array([cell.molecule_concs[self.metabolite]
                                          for cell in cells])
                                 if self.metabolite and all(hasattr(c, 'diffsys') and c.diffsys is not None for c in cells) else np.array([]))
            elif self.colorizer == Colorizer.LINEAGE_DISTANCE:
                # Assign colors based on lineage path using inferno-like colormap
                for cell in cells:
                    hsv_color = self._assign_lineage_color(cell)
                    rgb_color = self._hsv_to_rgb(hsv_color)
                    cell.recolor(rgb_color)
                return  # Skip the rest of the function since we've already colored the cells
            elif self.colorizer == Colorizer.LINEAGE:
                pass
            else:
                print(f"Error: Invalid colorizer type: {self.colorizer}")
                raise ValueError("Colorizer must be: PRESSURE, VOLUME, GENE, MOLECULE, LINEAGE_DISTANCE, or RANDOM.")

            if property_values is not None:
                values = self._scale(property_values)
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
    cell1: Cell, cell2: Cell, threshold=0.3
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
def _contact_areas(cells: list[Cell], threshold=5) -> tuple[dict, dict]:
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
    CELL_CONCENTRATIONS = auto()
    GENES = auto()
    GRID = auto() # larger overhead for each frame, so not saved by default

    DEFAULT = TIMES | DIVISIONS | MOTION_PATH | FORCE_PATH | VOLUMES | PRESSURES | CONTACT_AREAS | SHAPE_FEATURES | CELL_CONCENTRATIONS | GENES
    ALL = _all() # includes grid concentrations


class DataExporter(Handler):
    def __init__(self, path=None, options=DataFlag.DEFAULT):
        self.path = path
        self.h5file = None
        self.options = options

    def setup(self, get_cells, get_diffsystems, dt):
        super().setup(get_cells, get_diffsystems, dt)
        self.get_cells = get_cells
        self.get_diffsystems = get_diffsystems
        self.dt = dt
        self.time_start = datetime.now()

        # Use provided path or create one based on render filepath
        if self.path is None:
            render_path = bpy.context.scene.render.filepath
            render_dir = os.path.dirname(render_path)
            os.makedirs(render_dir, exist_ok=True)
            self.path = os.path.join(render_dir, "data.h5")

        if os.path.exists(self.path):
            os.remove(self.path)

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.h5file = h5py.File(self.path, 'w')
        self.h5file.attrs['seed'] = bpy.context.scene["seed"]

    def run(self, scene, depsgraph):
        frame_number = scene.frame_current
        frame_name = f"frame_{frame_number:03d}"
        frame_grp = self.h5file.create_group(frame_name)
        frame_grp.attrs["frame"] = frame_number
        frame_grp.attrs["time"] = (datetime.now() - self.time_start).total_seconds()

        cells_grp = frame_grp.create_group("cells")
        contact_areas = {}
        ratios = {}

        # Keep track of used names to handle duplicates
        used_names = set()

        for cell in self.get_cells():
            # Create a unique name for the cell group
            base_name = cell.name
            cell_name = base_name
            counter = 1
            while cell_name in used_names:
                cell_name = f"{base_name}_{counter}"
                counter += 1
            used_names.add(cell_name)

            try:
                cell_grp = cells_grp.create_group(cell_name)
                cell_grp.attrs["name"] = cell.name  # Store original name as attribute
                cell_grp.create_dataset("loc", data=np.array(cell.loc, dtype=np.float64))

                # Initialize all potential datasets with defaults
                if self.options & DataFlag.VOLUMES:
                    cell_grp.create_dataset("volume", data=np.nan)
                if self.options & DataFlag.PRESSURES:
                    cell_grp.create_dataset("pressure", data=np.nan)
                if self.options & DataFlag.DIVISIONS:
                    cell_grp.create_dataset("division_frame", data=np.nan)
                if self.options & DataFlag.FORCE_PATH:
                    cell_grp.create_dataset("force_loc", data=np.full(3, np.nan, dtype=np.float64))
                if self.options & DataFlag.SHAPE_FEATURES:
                    cell_grp.create_dataset("aspect_ratio", data=np.nan)
                    cell_grp.create_dataset("sphericity", data=np.nan)
                    cell_grp.create_dataset("compactness", data=np.nan)
                    cell_grp.create_dataset("sav_ratio", data=np.nan)

                # Populate datasets only if data is present
                if self.options & DataFlag.VOLUMES:
                    cell_grp["volume"][...] = float(cell.volume())
                if cell.physics_enabled and (self.options & DataFlag.PRESSURES):
                    cell_grp["pressure"][...] = float(cell.pressure)
                if self.options & DataFlag.DIVISIONS and getattr(cell, 'division_frame', None) is not None:
                    cell_grp["division_frame"][...] = float(cell.division_frame)
                if self.options & DataFlag.FORCE_PATH:
                    cell_grp["force_loc"][...] = np.array(cell.motion_force.loc, dtype=np.float64)

                if self.options & DataFlag.SHAPE_FEATURES:
                    cell_grp["aspect_ratio"][...] = float(cell.aspect_ratio())
                    cell_grp["sphericity"][...] = float(cell.sphericity())
                    cell_grp["compactness"][...] = float(cell.compactness())
                    cell_grp["sav_ratio"][...] = float(cell.sav_ratio())

                if self.options & DataFlag.GENES:
                    if hasattr(cell, 'gene_concs') and cell.gene_concs:
                        for gene, gene_conc in cell.gene_concs.items():
                            gene_name = gene.name if hasattr(gene, 'name') else str(gene)
                            cell_grp.create_dataset(f"gene_{gene_name}_conc", data=float(gene_conc))

                if hasattr(cell, 'molecule_concs') and cell.molecule_concs:
                    for mol, mol_conc in cell.molecule_concs.items():
                        mol_name = mol.name if hasattr(mol, 'name') else str(mol)
                        if isinstance(mol_conc, (tuple, list)):
                            if len(mol_conc) == 2 and isinstance(mol_conc[1], (int, float)):
                                cell_grp.create_dataset(f"mol_{mol_name}_conc", data=float(mol_conc[1]))
                            else:
                                cell_grp.create_dataset(f"mol_{mol_name}_conc", data=np.array(mol_conc, dtype=np.float64))
                        elif isinstance(mol_conc, (int, float)):
                            cell_grp.create_dataset(f"mol_{mol_name}_conc", data=float(mol_conc))
                        else:
                            print(f"Warning: Skipping molecule {mol_name} with unsupported concentration type: {type(mol_conc)}")
            except Exception as e:
                print(f"Warning: Could not save data for cell {cell.name}: {e}")
                continue

        # Grid concentration data
        if self.options & DataFlag.GRID:
            grid_grp = frame_grp.create_group("concentration_grid")
            for mol in self.get_diffsystem().molecules:
                mol_name = mol.name if hasattr(mol, 'name') else str(mol)
                mol_grp = grid_grp.create_group(mol_name)
                mol_grp.create_dataset("dimensions", data=np.array(self.get_diffsystem().grid_size, dtype=np.int32))
                mol_grp.create_dataset("values", data=np.array(self.get_diffsystem()._grid_concentrations[mol_name], dtype=np.float64))

        # Contact area and ratios
        if self.options & DataFlag.CONTACT_AREAS:
            contact_areas, ratios = _contact_areas(self.get_cells())
            # Create a single dataset for contact areas
            dt = np.dtype([
                ('source', h5py.string_dtype('utf-8')),
                ('target', h5py.string_dtype('utf-8')),
                ('area', float)
            ])
            data = []
            for source, contacts in contact_areas.items():
                for target, area in contacts:
                    data.append((str(source), str(target), area))  # Ensure strings

            area_data = np.array(data, dtype=dt)
            frame_grp.create_dataset("contact_areas", data=area_data)

    def close(self):
        if self.h5file:
            self.h5file.close()
            self.h5file = None

    def __del__(self):
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

    def points_to_volume_with_labels(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        grid_points = self.world_to_grid_coords(points).astype(int)

        # Clip to grid bounds
        valid_mask = np.all((grid_points >= 0) & (grid_points < self.resolution), axis=1)
        grid_points = grid_points[valid_mask]
        labels = labels[valid_mask]

        # Initialize label volume
        volume = np.zeros(self.resolution, dtype=np.uint8)

        # Assign each point's label (last one wins if overlap)
        volume[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]] = labels

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

        all_points = []
        all_labels = []
        for idx, obj in enumerate(visible_objects, start=1):  # Start at 1 to reserve 0 for background
            points = self.sample_mesh_points(obj)
            labels = np.full(len(points), idx, dtype=np.uint8)
            all_points.append(points)
            all_labels.append(labels)

        combined_points = np.vstack(all_points)
        combined_labels = np.concatenate(all_labels)

        # Combine all points
        combined_points = np.vstack(all_points)

        # Convert to volume
        volume = self.points_to_volume_with_labels(combined_points, combined_labels)

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
