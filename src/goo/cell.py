from typing import ClassVar

import bmesh
import bpy
import numpy as np

from bpy.types import ClothModifier, CollisionModifier
from mathutils import Vector

from .force import *
from .gene import (
    Circuit,
    Gene,
    GeneRegulatoryNetwork as GRN,
)
from .growth import *
from .molecule import DiffusionSystem, Molecule
from .utils import *


class Cell(BlenderObject):
    """A cell.

    Cells are represented in Blender by mesh objects. They can interact with
    physics by adding Blender modifiers, and the forces that influence its
    motion is determined by an associated collection.

    Args:
        obj: Blender object to be used as the representation of the cell.

    Attributes:
        celltype (CellType): The cell type to which the cell belongs.
        homo_adhesion_strength (float): The homotypic adhesion strength for this cell.
            If set, overrides the cell type's default adhesion strength.
    """

    def __init__(self):
        self._name = ""
        self.obj: bpy.types.Object = None
        self._color: tuple[float, float, float] = None
        self._homo_adhesion_strength: float = None

        self.direction = Vector()
        self.adhesion_force: AdhesionForce = None
        self.adhesion_forces: list[AdhesionForce] = []
        self.motion_force: MotionForce = None
        self.effectors: ForceCollection = None

        self.growth_controller: GrowthController = None
        self.grn: GRN = None
        self.diffsys: DiffusionSystem = None
        self.grn_hooks = []
        self.diffsys_hooks = []

        self.just_divided = False
        self.division_frame = None
        self.physics_enabled = False
        self.mod_settings = []

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        if self.obj:
            self.obj.name = name
        if self.effectors:
            self.effectors.name = name + "_effectors"
        if self.motion_force:
            self.motion_force.name = name + "_motion"

    @property
    def mat(self):
        if self.obj.data.materials:
            return self.obj.data.materials[0]

        return None

    @property
    def homo_adhesion_strength(self) -> float:
        """Get the homotypic adhesion strength for this cell.
        If not set, returns the cell type's default adhesion strength.
        If cell type is not set, returns None.
        """
        if self._homo_adhesion_strength is not None:
            return self._homo_adhesion_strength
        if hasattr(self, 'celltype'):
            return self.celltype.homo_adhesion_strength
        return None

    @homo_adhesion_strength.setter
    def homo_adhesion_strength(self, strength: float):
        """Set the homotypic adhesion strength for this cell.
        If set to None, will use the cell type's default adhesion strength.
        """
        self._homo_adhesion_strength = strength
        # If physics is enabled, update the adhesion force
        if self.physics_enabled and self.adhesion_force:
            self.adhesion_force.strength = strength

    def copy(self):
        other_obj = self.obj.copy()
        other_obj.data = self.obj.data.copy()
        other = self.celltype.create_cell(self.name + "_copy", self.loc, obj=other_obj)

        other.growth_controller = self.growth_controller.copy()
        other.grn = self.grn.copy()

        other.physics_enabled = self.physics_enabled
        if self.cloth_mod:
            other.pressure = self.pressure
            other.stiffness = self.stiffness
            other._update_cloth()
        other.mod_settings = self.mod_settings.copy()

        return other

    # ===== Mesh properties =====
    def volume(self) -> float:
        """Calculates the volume of the cell.

        Returns:
            The signed volume of the cell (with physics evaluated).
        """
        bm = bmesh.new()
        bm.from_mesh(self.obj_eval.to_mesh())
        bm.transform(self.obj_eval.matrix_world)
        volume = bm.calc_volume()
        bm.free()
        return volume

    # TODO: this is scale invariant! (does not update with scale)
    def area(self) -> float:
        """Calculates the surface area of the cell.

        Returns:
            The surface area of the cell.
        """
        faces = self.obj_eval.data.polygons
        area = sum(f.area for f in faces)
        return area

    def COM(self, local_coords: bool = False) -> Vector:
        """Calculates the center of mass of the cell.

        Args:
            local_coords: if `True`, coordinates are returned in local object
                space rather than world space.

        Returns:
            The vector representing the center of mass of the cell.
        """
        vert_coords = self.vertices(local_coords)
        if len(vert_coords) == 0:
            # If no vertices, return the cell's current location
            return self.loc if not local_coords else Vector((0, 0, 0))
        com = Vector(np.mean(vert_coords, axis=0))
        return com

    def aspect_ratio(self) -> float:
        """Calculates the aspect ratio of the cell.

        The aspect ratio is the ratio of the major axis to the minor axis of the cell.

        Returns:
            The aspect ration value for a cell.
        """
        try:
            major_axis = self.major_axis().length()
            minor_axis = self.minor_axis().length()
            aspect_ratio = major_axis / minor_axis
            return aspect_ratio
        except (ValueError, ZeroDivisionError, AttributeError):
            # Return 1.0 (perfect sphere) if we can't calculate the aspect ratio
            return 1.0

    def sphericity(self) -> float:
        """Calculates the sphericity of the cell.

        The sphericity is a measure of how closely a cell resembles a perfect sphere.
        It is calculated as the ratio of the surface area of a sphere with the same volume
        as the cell to the surface area of the cell.

        Returns:
            The sphericity value for the cell.
        """

        volume = self.volume()
        surface_area = self.area()
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
        return sphericity

    def compactness(self) -> float:
        """Calculates the compactness of the cell.

        Compactness provides a measure of how efficiently the volume
        is enclosed by the surface area, calculated as .

        Returns:
            The compactness value for the cell.
        """

        volume = self.volume()
        area = self.area()
        compactness = volume**2 / area**3
        return compactness

    def sav_ratio(self) -> float:
        """Calculates the surface area: volume (SA:V) ratio of the cell.

        Returns:
            The SA:V ratio of the cell.
        """

        volume = self.volume()
        area = self.area()
        sav_ratio = area / volume
        return sav_ratio

    def _get_eigenvector(self, n: int) -> Axis:
        """Returns the nth eigenvector (axis) in object space as a line defined
        by two Vectors.

        This function calculates the nth eigenvector of the covariance matrix
        defined by the cell's vertices. The eigenvector axis is defined by two
        points: the vertices in the direction of the eigenvector with the
        smallest and largest projections.

        Args:
            n: The index of the eigenvector to return.

        Returns:
            An axis defined by the eigenvector and the vertices at the
            extreme projections along this vector.
        """
        verts = self.vertices()

        if len(verts) < 2:
            # If not enough vertices, return a default axis
            default_axis = Vector((1, 0, 0) if n == 0 else (0, 1, 0) if n == 1 else (0, 0, 1))
            return Axis(default_axis, Vector((0, 0, 0)), Vector((1, 0, 0)), self.obj_eval.matrix_world)

        try:
            # Calculate the eigenvectors and eigenvalues of the covariance matrix
            covariance_matrix = np.cov(verts, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
            axis = eigenvectors[:, n]

            projections = verts @ axis
            min_index = np.argmin(projections)
            max_index = np.argmax(projections)
            first_vertex = Vector(verts[min_index])
            last_vertex = Vector(verts[max_index])
            return Axis(Vector(axis), first_vertex, last_vertex, self.obj_eval.matrix_world)
        except (ValueError, np.linalg.LinAlgError):
            # If eigen decomposition fails, return a default axis
            default_axis = Vector((1, 0, 0) if n == 0 else (0, 1, 0) if n == 1 else (0, 0, 1))
            return Axis(default_axis, Vector((0, 0, 0)), Vector((1, 0, 0)), self.obj_eval.matrix_world)

    def major_axis(self) -> Axis:
        """Returns the major axis of the cell."""
        return self._get_eigenvector(0)

    def minor_axis(self) -> Axis:
        """Returns the minor axis of the cell."""
        return self._get_eigenvector(1)

    def radius(self):
        """Calculate the radius based on the major and minor axes of the cell."""
        major_axis = self.major_axis()
        minor_axis = self.minor_axis()
        return (major_axis.length() + minor_axis.length()) / 2

    # ===== Mesh operations =====
    @property
    def obj_eval(self) -> bpy.types.ID:
        """The evaluated object.

        Note:
            See the `Blender API Documentation for evaluated_get(depsgraph)
            <https://docs.blender.org/api/current/bpy.types.ID.html?highlight=evaluated_get#bpy.types.ID.evaluated_get>`__.
        """
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = self.obj.evaluated_get(dg)
        return obj_eval

    def vertices(self, local_coords: bool = False) -> list[Vector]:
        """Returns the vertices of the mesh representation of the cell.

        Args:
            local_coords: if `True`, coordinates are returned in local object space
            rather than world space.

        Returns:
            List of coordinates of vertices.
        """
        try:
            verts = self.obj_eval.data.vertices
            if len(verts) == 0:
                print(f"Warning: Cell {self.name} has no vertices - this may indicate a collapsed cell")
                return []
            if local_coords:
                return [v.co.copy() for v in verts]
            else:
                matrix_world = self.obj_eval.matrix_world
                return [matrix_world @ v.co for v in verts]
        except (AttributeError, RuntimeError) as e:
            print(f"Warning: Error accessing vertices for cell {self.name}: {e!s}")
            return []

    def is_collapsed(self) -> bool:
        """Checks if the cell has collapsed or is in an invalid state.

        A cell is considered collapsed if:
        1. It has no vertices
        2. Its volume is zero or negative
        3. Its mesh is invalid

        Returns:
            bool: True if the cell is collapsed, False otherwise
        """
        try:
            # Check for zero vertices
            if len(self.vertices()) == 0:
                return True

            # Check for invalid volume
            volume = self.volume()
            if volume <= 0:
                return True

            # Check for invalid mesh
            if not self.obj_eval or not self.obj_eval.data or not self.obj_eval.data.vertices:
                return True

            return False
        except Exception as e:
            print(f"Warning: Error checking if cell {self.name} is collapsed: {e!s}")
            return True

    def recenter(self, origin=True, forces=True):
        """Recenter cell origin to center of mass of cell,
        and center forces to that same origin."""
        com = self.COM()

        # Recenter mesh origin to COM
        bm = bmesh.new()
        bm.from_mesh(self.obj.to_mesh())
        bmesh.ops.translate(bm, verts=bm.verts, vec=-self.COM(local_coords=True))
        bm.to_mesh(self.obj.data)
        bm.free()
        self.loc = com

        # Recenter forces to COM
        for force in self.adhesion_forces:
            force.loc = self.loc

    def remesh(self, voxel_size: float = 0.7, smooth: bool = True) -> None:
        """Remesh the underlying mesh representation of the cell.

        Remeshing is done using the built-in `voxel_remesh()`.

        Args:
            voxel_size: The resolution used for the remesher (smaller means more
            polygons).  smooth: If true, the final cell faces will appear smooth.

        Note:
            The `voxel_remesh()` operator is used to remesh the cell. This operator
            is faster than using the remesh modifier, but it can only be used on
            objects with a mesh data type.
        """
        # use of object ops is 2x faster than remeshing with modifiers
        self.obj.data.remesh_mode = "VOXEL"
        self.obj.data.remesh_voxel_size = voxel_size
        with bpy.context.temp_override(active_object=self.obj, object=self.obj):
            bpy.ops.object.voxel_remesh()

        for f in self.obj.data.polygons:
            f.use_smooth = smooth

    @property
    def color(self) -> tuple[float, float, float]:
        """Color of the cell"""
        return self.mat.diffuse_color[:3]

    @color.setter
    def color(self, color: tuple[float, float, float]) -> None:
        """Sets the color of the cell.

        Args:
            color: A tuple (r, g, b) representing the new color to apply.
        """
        self.mat.diffuse_color[:3] = color

    def recolor(self, color: tuple[float, float, float]) -> None:
        """Recolors the material of the cell.

        This function changes the diffuse color of the cell's material to the
        specified color while preserving the alpha value. If the material uses
        nodes, it also updates the 'Base Color' input of any nodes that have it.

        Args:
            color: A tuple (r, g, b) representing the new color to apply.
        """
        if not self.mat:
            print(f"Warning: Cell {self.name} has no material!")
            return

        # If material is shared (used by multiple objects), make a unique copy
        if self.mat.users > 1:
            new_mat = self.mat.copy()  # Duplicate the material
            new_mat.name = f"{self.name}_mat"  # Give it a unique name
            self.obj.data.materials[0] = new_mat  # Assign the new material to the cell

        # Apply the color
        r, g, b = color
        a = 1.0  # Ensure full opacity
        self.mat.diffuse_color = (r, g, b, a)

        # If using material nodes, update Base Color in the node tree
        if self.mat.use_nodes:
            for node in self.mat.node_tree.nodes:
                if "Base Color" in node.inputs:
                    node.inputs["Base Color"].default_value = (r, g, b, a)

    # ===== Physics operations =====
    def enable_physics(self):
        """Enable the physics simulation for the cell.

        This function re-enables the physics simulation for the cell by recreating
        the modifier stack from stored settings, updating the cloth modifier,
        and enabling any adhesion forces.

        Raises:
            RuntimeError: If physics is already enabled.
        """
        if self.physics_enabled:
            raise RuntimeError(
                f"Trying to enable physics on cell {self.name} when already enabled!"
            )

        # recreate modifier stack
        for name, type, settings in self.mod_settings:
            mod = self.obj.modifiers.new(name=name, type=type)
            declare_settings(mod, settings)
        self.mod_settings.clear()

        # ensure cloth mod is set correctly
        self._update_cloth()
        if self.cloth_mod:
            self.cloth_mod.point_cache.frame_end = bpy.context.scene.frame_end

        for force in self.adhesion_forces:
            force.enable()
        self.physics_enabled = True

    def disable_physics(self):
        """
        Disable the physics simulation for the cell.

        This function disables the physics simulation for the cell by storing the
        current modifier settings, removing all modifiers, and disabling any adhesion
        forces.

        Note:
            If physics is already disabled, this function will do nothing.
        """
        if not self.physics_enabled:
            return  # Exit early if physics is already disabled

        for mod in self.obj.modifiers:
            name, type = mod.name, mod.type
            settings = store_settings(mod)
            self.mod_settings.append((name, type, settings))
            self.obj.modifiers.remove(mod)

        for force in self.adhesion_forces:
            force.disable()
        self.physics_enabled = False

    def _update_cloth(self):
        """Update the cloth modifier is correctly set to be affected by forces
        acting upon the cell.
        """
        if self.cloth_mod and self.effectors:
            self.cloth_mod.settings.effector_weights.collection = self.effectors.col

    def get_modifier(self, type) -> bpy.types.Modifier | None:
        """Retrieves the first modifier of the specified type from the
        underlying object representation of the cell.

        Args:
            type: The type of the modifier to search for.

        Returns:
            The first modifier of the specified type if found, otherwise None.
        """
        return next((m for m in self.obj.modifiers if m.type == type), None)

    @property
    def cloth_mod(self) -> ClothModifier | None:
        """The cloth modifier of the cell if it exists, otherwise None."""
        return self.get_modifier("CLOTH")

    @property
    def collision_mod(self) -> CollisionModifier | None:
        """The collision modifier of the cell if it exists, otherwise None."""
        return self.get_modifier("COLLISION")

    @property
    def stiffness(self) -> float:
        """Stiffness of the membrane of the cell."""
        return self.cloth_mod.settings.tension_stiffness

    @stiffness.setter
    def stiffness(self, stiffness: float):
        self.cloth_mod.settings.tension_stiffness = stiffness
        self.cloth_mod.settings.compression_stiffness = stiffness
        self.cloth_mod.settings.shear_stiffness = stiffness

    @property
    def pressure(self) -> float:
        """Internal pressure of the cell."""
        return self.cloth_mod.settings.uniform_pressure_force

    @pressure.setter
    def pressure(self, pressure: float):
        self.cloth_mod.settings.uniform_pressure_force = pressure
        if self.growth_controller:
            self.growth_controller.set_pressure(pressure)

    # ===== Force operations =====
    def add_effector(self, force: Force | ForceCollection):
        """Add a force or a collection of forces that affects this cell.

        Args:
            force: The force or collection of forces to add.
        """
        # If effectors is not yet instantiated, create new effectors.
        if self.effectors is None:
            self.effectors = ForceCollection(f"{self.name}_effectors")
            self.effectors.show()
        self.effectors.add(force)

    def remove_effector(self, force: Force | ForceCollection):
        """Remove a force or a collection of forces that affects this cell.

        Args:
            force: The force or collection of forces to remove.
        """
        self.effectors.remove(force)

    def add_force(self, force: Force):
        """Add a force that affects this cell.

        Args:
            force: The force to add.
        """
        if isinstance(force, AdhesionForce):
            self.adhesion_forces.append(force)
        elif isinstance(force, MotionForce):
            self.add_effector(force)
            self.motion_force = force

    # ===== Cell actions =====
    def divide(self, division_logic):
        """Divide the cell into two daughter cells."""
        mother, daughter = division_logic.make_divide(self)
        mother.just_divided = True
        daughter.just_divided = True
        mother.division_frame = bpy.context.scene.frame_current
        daughter.division_frame = bpy.context.scene.frame_current

        return mother, daughter

    def move(self, direction: tuple | None = None, strength=None):
        """Set move location. If direction is not specified, then use previous
        direction as reference.
        """
        if direction is not None:
            self.direction = Vector(direction)
        elif self.direction is None:
            raise ValueError(
                "Direction must be specified if cell's direction \
                    has not been previously set!"
            )

        motion_loc = self.loc + self.direction.normalized() * (2 + self.radius())

        self.motion_force.loc = motion_loc
        self.motion_force.point_towards(self.loc)
        if strength is not None:
            self.motion_force.strength = strength

    def step_growth(self, dt=1):
        if self.just_divided:
            self.growth_controller.step_divided(self.volume())
            self.just_divided = False
        else:
            new_pressure = self.growth_controller.step_growth(self.volume(), dt)
            self.pressure = new_pressure

    # ===== Gene regulatory network items =====
    def step_grn(self, dt=1):
        """Calculate and update the gene concentrations
        of the gene regulatory network after 1 time step."""
        com = self.COM()
        radius = self.radius()

        # Step through GRN
        self.grn.update_concs(com, radius, dt)
        self["genes"] = {str(k): v for k, v in self.gene_concs.items()}

    def update_physics_with_grn(self):
        # Update physical attributes
        for hook in self.grn_hooks:
            hook()

    @property
    def gene_concs(self):
        return self.grn.gene_concs

    @gene_concs.setter
    def gene_concs(self, gene_concs):
        self["genes"] = {str(k): v for k, v in gene_concs.items()}
        self.grn.gene_concs = gene_concs

    def link_gene_to_property(self, gene: Gene, property, gscale=(0, 1), pscale=(0, 1)):
        """Link gene to property, so that changes in the gene levels
        will cause changes in the physical property.
        """
        if property == "motion_direction":
            hook = create_direction_updater(self, gene)
        if property == "motion_strength":
            hook = create_motion_strength_updater(self, gene)
        else:
            hook = create_scalar_updater(self, gene, property, "linear", gscale, pscale)

        self.grn_hooks.append(hook)

    def __setitem__(self, name: str, value) -> None:
        self.obj[name] = value

    def __getitem__(self, name: str):
        return self.obj[name]

    def __contains__(self, k: str):
        return self.obj.__contains__(k)

    # ===== Sensing and secretion =====
    @property
    def molecule_concs(self):
        """Get the concentrations of the molecules around the cell radius."""
        if not self.diffsys:
            return {}
        return {mol: self.diffsys.get_total_ball_concentrations(mol, self.COM(), self.radius()) for mol in self.diffsys.molecules}

    def link_molecule_to_property(self, molecule: Molecule, property):
        """Link molecule to property, so that changes in the extracellular
        concentrations of the molecule will cause changes in the physical property.
        """
        if property == "motion_direction":
            hook = create_direction_updater(self, molecule)
        else:
            hook = create_scalar_updater(self, molecule, property, "linear")

        self.diffsys_hooks.append(hook)

    def sense(self, mol: Molecule):
        """Step the sensing of the cell."""
        coords, concentrations = self.diffsys.get_ball_concentrations(
            mol=mol,
            center=self.COM(),
            radius=self.radius(),
        )
        return coords, concentrations

    def secrete(self, mol: Molecule):
        """Step the secretion of the cell."""
        self.diffsys.update_concentrations_around_sphere(
            mol=mol,
            center=self.COM(),
            radius=self.radius(),
            value=0.5,
        )

    def update_physics_with_molecules(self, mol: Molecule):
        """Step the extracellular molecules of the cell."""
        # update physical attributes
        for hook in self.diffsys_hooks:
            hook()


class PropertyUpdater:
    """A class that updates a property of a cell based on a gene value.

    Args:
        getter: The getter function.
        transformer: The transformer function.
        setter: The setter function.

    Attributes:
        getter: A function that retrieves the gene value from a diffusion system.
        transformer: A function that transforms the gene value into a property value.
        setter: A function that sets the property value of a cell.
    """
    def __init__(self, getter, transformer, setter):
        self.getter = getter
        self.transformer = transformer
        self.setter = setter

    def __call__(self):
        """Update the property of a cell based on the gene value."""
        metabolite_value = self.getter()
        prop_value = self.transformer(metabolite_value)
        self.setter(prop_value)


def create_scalar_updater(
    cell: Cell,
    metabolite: Gene | Molecule,
    property,
    relationship="linear",
    gscale=(0, 1),
    pscale=(0, 1),
):
    """Create a scalar updater that links a gene to a scalar property of a cell.

    Args:
        cell: The cell object.
        gene: The gene that is linked to the property.
        property: The property that is linked to the gene.
        relationship: The relationship between the gene and the property.
        gscale: The scale of the gene values.
        pscale: The scale of the property values.

    Returns:
        A PropertyUpdater object that can be used to update the property.
    """
    gmin, gmax = gscale
    pmin, pmax = pscale

    # Simple gene getter
    def metabolite_getter():
        """Get the gene value from the cell's gene regulatory network."""
        if isinstance(metabolite, Molecule):
            return cell.sense(mol=metabolite)
        else:
            return cell.gene_concs[metabolite]

    # Different transformers
    def linear_transformer(gene_value):
        """Transform the gene value into a property value using a linear relationship."""
        if gene_value <= gmin:
            return pmin
        if gene_value >= gmax:
            return pmax
        return (gene_value - gmin) / (gmax - gmin) * (pmax - pmin) + pmin

    # Different property setters
    def set_motion_force(prop_value):
        """Set the motion force of the cell."""
        cell.motion_force = prop_value

    match relationship:
        case "linear":
            transformer = linear_transformer
        case _:
            raise NotImplementedError()

    match property:
        case "motion_force":
            setter = set_motion_force
        case _:
            raise NotImplementedError()

    return PropertyUpdater(metabolite_getter, transformer, setter)


def create_direction_updater(
    cell: Cell,
    metabolite: Gene | Molecule,
):
    """Create a direction updater that links a gene or molecule to the direction of a cell.

    Args:
        cell: The cell object.
        metabolite: The gene or molecule that is linked to the direction.
    """
    def metabolite_conc_getter():
        """Get the molecule values from the cell's gene regulatory network or diffusion system."""
        if isinstance(metabolite, Molecule):
            return cell.diffsys.get_ball_concentrations(
                mol=metabolite,
                center=cell.COM(),
                radius=cell.radius()
            )
        else:
            return cell.gene_concs[metabolite]

    def weighted_direction(metabolite_conc):
        """Calculate the weighted direction of the cell based on the molecule values."""
        coords, concs = metabolite_conc
        direction = np.average(coords, axis=0, weights=concs)
        return direction - cell.loc

    def set_direction(direction):
        """Set the direction of the cell."""
        cell.move(direction)

    return PropertyUpdater(metabolite_conc_getter, weighted_direction, set_direction)


def create_motion_strength_updater(
        cell: Cell,
        metabolite: Gene | Molecule,
        gscale=(0.5, 2),
        pscale=(0, 1),
):
    """Create a motion strength updater that links a gene to
    the motion strength of a cell.

    Args:
        cell: The cell object.
        gene: The gene that is linked to the motion strength.
        gscale: The scale of the gene values.
        pscale: The scale of the motion strength values.

    Returns:
        A PropertyUpdater object that can be used to update the motion strength.
    """

    gmin, gmax = gscale
    pmin, pmax = pscale

    def metabolite_conc_getter():
        """Get the gene concentration from the cell's gene regulatory network."""
        metabolite_conc = cell.gene_concs.get(metabolite)
        normalized_metabolite_conc = linear_transformer(metabolite_conc)
        return normalized_metabolite_conc

    def linear_transformer(metabolite_value):
        """Normalize the gene value to a value between gscale min and max."""
        if metabolite_value <= gmin:
            return pmin
        if metabolite_value >= gmax:
            return pmax
        return (metabolite_value - gmin) / (gmax - gmin) * (pmax - pmin) + pmin

    def weighted_motion_strength(metabolite_value):
        """Calculate the weighted motion strength of the cell based on the gene value."""
        motion_strength = (metabolite_value * cell.motion_force.init_strength)
        return motion_strength

    def set_motion_strength(motion_strength):
        """Set the motion strength of the cell."""
        cell.celltype.motion_strength = motion_strength

    return PropertyUpdater(
        metabolite_conc_getter,
        weighted_motion_strength,
        set_motion_strength
        )


def store_settings(mod: bpy.types.bpy_struct) -> dict:
    """Store the settings of a Blender modifier in a dictionary.

    Args:
        mod: The Blender modifier.

    Returns:
        A dictionary with the stored settings.
    """

    settings = {}
    for p in mod.bl_rna.properties:
        id = p.identifier
        if not p.is_readonly:
            settings[id] = getattr(mod, id)
        elif id in ["settings", "collision_settings", "effector_weights"]:
            settings[id] = store_settings(getattr(mod, id))
    return settings


def declare_settings(mod: bpy.types.bpy_struct, settings: dict, path="mod"):
    for id, setting in settings.items():
        try:
            if isinstance(setting, dict):
                submod = getattr(mod, id, None)
                if submod is not None:
                    declare_settings(submod, setting, path=f"{path}.{id}")
                else:
                    print(f"Warning: {path} has no attribute '{id}'")
            else:
                prop = mod.bl_rna.properties.get(id)
                if prop is not None:
                    if hasattr(prop, 'array_length') and isinstance(setting, list | tuple):
                        expected_len = prop.array_length
                        if len(setting) == expected_len:
                            try:
                                # For vector properties, ensure we have the right type
                                if prop.type == 'FLOAT' and prop.subtype == 'XYZ':
                                    # For XYZ vectors, use Vector type
                                    coerced = Vector(setting)
                                else:
                                    # For other array types, use tuple
                                    coerced = tuple(float(x) for x in setting)
                                setattr(mod, id, coerced)
                            except Exception as e:
                                print(f"Warning: Could not coerce {path}.{id} to appropriate type: {e}")
                        else:
                            print(f"Warning: {path}.{id} expected length {expected_len}, got {len(setting)}")
                    else:
                        setattr(mod, id, setting)
                else:
                    setattr(mod, id, setting)
        except Exception as e:
            print(f"Warning: Could not set {path}.{id}: {e}")


class CellType:
    """A cell type.

    Cell types are represented in Blender by collection. Cells of the same cell
    type interact through homotypic adhesion forces, and interact with cells of
    different cell types through heterotypic adhesion forces. Cell types have
    default mesh creation settings (including color) and default physics
    settings (including cloth, collision, homotypic adhesion forces, and motion
    forces).

    Args:
        name: The name of the cell type.
        physics_enabled: Whether cells of this cell type are responsive to physics.

    Attributes:
        homo_adhesion_strength (int): Default homotypic adhesion strength
            between cells of this type.
        motion_strength (int): Default motion strength of
            cells of this type.
    """

    _default_celltype = None

    def __init__(
        self,
        name,
        pattern="standard",
        target_volume=30,
        homo_adhesion_strength=2000,
        hetero_adhesion_strengths={},
        motion_strength=0,
        **kwargs,
    ):
        self.name = name
        self.cells = []

        if type(pattern) is str:
            match pattern:
                case "standard":
                    self.pattern = StandardPattern()
                case "simple":
                    self.pattern = SimplePattern()
                case "yolk":
                    self.pattern = YolkPattern()
                case _:  # default to simple
                    self.pattern = SimplePattern()
        else:
            self.pattern = pattern
        self.target_volume = target_volume
        self.kwargs = kwargs

        self.homo_adhesion_strength = homo_adhesion_strength
        self._homo_adhesion_collection = ForceCollection(name)

        self.motion_strength = motion_strength

        self.hetero_adhesion_strengths = {}
        self._hetero_adhesion_collections = {}
        for celltype, strength in hetero_adhesion_strengths.items():
            self.set_hetero_adhesion_strength(celltype, strength)

        # Store molecule-property links for this cell type
        self._molecule_property_links = []

        # Store diffusion system for this cell type
        self._diffsys = None

    @staticmethod
    def default_celltype() -> "CellType":
        """Get the default cell type."""
        if CellType._default_celltype is None:
            CellType._default_celltype = CellType("default")
        return CellType._default_celltype

    @property
    def diffsys(self):
        """Get the diffusion system for this cell type."""
        return self._diffsys

    @diffsys.setter
    def diffsys(self, diffsys):
        """Set the diffusion system for this cell type and apply it to all cells.

        Args:
            diffsys: The diffusion system to set
        """
        self._diffsys = diffsys

        # Apply to all existing cells
        for cell in self.cells:
            cell.diffsys = diffsys

            # Re-apply molecule-property links since they need the diffusion system
            if hasattr(self, '_molecule_property_links'):
                for molecule, property in self._molecule_property_links:
                    cell.link_molecule_to_property(molecule, property)

    def create_cell(
        self,
        name,
        loc,
        color: tuple | None = None,
        physics_enabled: bool = True,
        physics_constructor: PhysicsConstructor = None,
        growth_enabled: bool = True,
        growth_type: GrowthType = GrowthType.LINEAR,
        growth_rate: float = 1,
        target_volume: float | None = None,
        genes_enabled: bool = True,
        circuits: list[Circuit] | None = None,
        genes: dict[str, float] = {},
        obj: bpy.types.Object = None,
        **mesh_kwargs,
    ):
        mesh_kwargs.update(self.kwargs)  # override with settings
        if obj:
            self.pattern.set_obj(name, obj)
        else:
            self.pattern.build_obj(name, loc, color=color, **mesh_kwargs)

        if physics_enabled:
            self.pattern.build_forces(
                self.homo_adhesion_strength,
                self._homo_adhesion_collection,
                self.motion_strength,
                self.hetero_adhesion_strengths,
                self._hetero_adhesion_collections,
            )
            # Building physics after forces ensures that the cloth modifier
            # can properly see the target effector collection.
            if not obj:
                self.pattern.build_physics(
                    physics_constructor=physics_constructor,
                )

        if growth_enabled:
            self.pattern.build_growth_controller(
                growth_type=growth_type,
                growth_rate=growth_rate,
                target_volume=target_volume if target_volume else self.target_volume,
            )
        if genes_enabled:
            self.pattern.build_network(
                circuits=circuits,
                genes=genes,
            )
        cell = self.pattern.retrieve_cell()
        cell.celltype = self
        cell.remesh()

        # Apply diffusion system if one exists for this cell type
        if self._diffsys is not None:
            cell.diffsys = self._diffsys

        # Apply any stored molecule-property links to the new cell
        for molecule, property in self._molecule_property_links:
            if hasattr(cell, 'diffsys') and cell.diffsys is not None:
                cell.link_molecule_to_property(molecule, property)

        self.cells.append(cell)
        return cell

    def set_hetero_adhesion_strength(self, other_celltype: "CellType", strength: float):
        outgoing_collections = ForceCollection(f"{self.name}_to_{other_celltype.name}")
        incoming_collections = ForceCollection(f"{other_celltype.name}_to_{self.name}")

        self.hetero_adhesion_strengths[other_celltype] = strength
        self._hetero_adhesion_collections[other_celltype] = (
            outgoing_collections,
            incoming_collections,
        )

        other_celltype.hetero_adhesion_strengths[self] = strength
        other_celltype._hetero_adhesion_collections[self] = (
            incoming_collections,
            outgoing_collections,
        )

    def link_molecule_to_property(self, molecule: Molecule, property):
        """Link molecule to property for all cells of this type.

        This will set up the link for all existing cells and any new cells created
        from this cell type.

        Args:
            molecule: The molecule to link
            property: The property to link the molecule to
        """
        # Store the link for future cells
        self._molecule_property_links.append((molecule, property))

        # Apply the link to all existing cells
        for cell in self.cells:
            if hasattr(cell, 'diffsys') and cell.diffsys is not None:
                cell.link_molecule_to_property(molecule, property)


class CellPattern:
    """Builders of different cell parts. The CellType class takes these
    patterns in order to create new cells."""

    mesh_kwargs: ClassVar[dict] = {}
    color: ClassVar[tuple | None] = None
    physics_constructor: ClassVar[PhysicsConstructor | None] = None
    initial_pressure: ClassVar[float | None] = None

    def __init__(self):
        self.reset()
        self.circuits = []
        self.genes = {}

    def reset(self):
        self._cell = Cell()

    def retrieve_cell(self):
        """Retrieves constructed cell, links cell to the scene,
        and resets the director.
        """
        # Link cell to scene
        cell = self._cell
        bpy.context.scene.collection.objects.link(cell.obj)

        self.reset()
        return cell

    @staticmethod
    def _override(base_option, user_option):
        if user_option is None:
            return base_option
        if isinstance(base_option, dict):
            return dict(base_option, **user_option)
        return user_option

    def set_obj(self, name, obj):
        self._cell.name = name
        self._cell.obj = obj

    def build_obj(
        self,
        name,
        loc,
        color=None,
        **mesh_kwargs,
    ):
        self._cell.name = name
        mesh_kwargs = self._override(self.__class__.mesh_kwargs, mesh_kwargs)
        color = self._override(self.__class__.color, color)

        # Create cell object
        obj = create_mesh(name, loc, mesh="icosphere", **mesh_kwargs)
        self._cell.obj = obj

        if color is not None:
            mat = create_material(f"{name}_material", color=color)
            obj.data.materials.append(mat)

    def build_physics(
        self,
        physics_constructor=None,
    ):
        # Add physics modifiers to cell object
        physics_constructor = self._override(
            self.__class__.physics_constructor, physics_constructor
        )
        if physics_constructor is not None:
            physics_constructor(self._cell)
            if hasattr(self._cell, 'cloth_mod') and self._cell.cloth_mod:
                self._cell.cloth_mod.point_cache.frame_end = bpy.context.scene.frame_end
            self._cell.enable_physics()

    def build_forces(
        self,
        homo_adhesion_strength: int,
        homo_adhesion_collection: ForceCollection,
        motion_strength: int,
        hetero_adhesion_strengths: dict[CellType, int],
        hetero_adhesion_collections: dict[CellType, tuple[ForceCollection]],
    ):
        # Use cell's adhesion strength if set, otherwise use provided homo_adhesion_strength
        strength = self._cell.homo_adhesion_strength
        if strength is None:
            strength = homo_adhesion_strength
        homo_adhesion = create_adhesion(strength, obj=self._cell.obj)
        homo_adhesion_collection.add(homo_adhesion)

        self._cell.add_force(homo_adhesion)
        self._cell.adhesion_force = homo_adhesion
        self._cell.add_effector(homo_adhesion_collection)

        for celltype in hetero_adhesion_strengths.keys():
            strength = hetero_adhesion_strengths[celltype]
            incoming, outgoing = hetero_adhesion_collections[celltype]

            hetero_adhesion = create_adhesion(
                strength,
                name=f"{self._cell.name}_to_{celltype.name}",
                loc=self._cell.loc,
            )
            outgoing.add(hetero_adhesion)

            self._cell.add_force(hetero_adhesion)
            self._cell.add_effector(incoming)

        motion_force = create_motion(
            name=f"{self._cell.name}_motion",
            loc=self._cell.loc,
            strength=motion_strength,
        )
        self._cell.add_force(motion_force)

    def build_growth_controller(
        self, growth_type: GrowthType, growth_rate: float = 1, target_volume=30
    ):
        controller = PIDController(
            self._cell.volume(),
            growth_type=growth_type,
            growth_rate=growth_rate,
            target_volume=target_volume,
        )
        if self.__class__.initial_pressure is not None:
            controller.initial_pressure = self.__class__.initial_pressure
            controller.previous_pressure = self.__class__.initial_pressure

        self._cell.pressure = controller.initial_pressure
        self._cell.growth_controller = controller

    def build_network(self, circuits=None, genes=None):
        circuits = [] if circuits is None else list(circuits)
        genes = {} if genes is None else dict(genes)

        grn = GRN()
        grn.load_circuits(*circuits)
        self._cell.grn = grn
        self._cell.genes = genes


class StandardPattern(CellPattern):
    mesh_kwargs: ClassVar[dict] = {}
    physics_constructor: ClassVar[PhysicsConstructor] = PhysicsConstructor(
        SubsurfConstructor,
        ClothConstructor,
        CollisionConstructor,
        RemeshConstructor,
    )
    color: ClassVar[tuple[float, float, float]] = (0.007, 0.021, 0.3)
    initial_pressure: ClassVar[float] = 12


class SimplePattern(CellPattern):
    """A cell type that is reduced in complexity for faster rendering."""

    mesh_kwargs: ClassVar[dict] = {}
    physics_constructor: ClassVar[PhysicsConstructor] = PhysicsConstructor(
        SimpleClothConstructor,
        CollisionConstructor,
    )
    color: ClassVar[tuple[float, float, float]] = (0.5, 0.5, 0.5)


class YolkPattern(CellPattern):
    """A larger cell type used for modeling embryonic yolks."""

    mesh_kwargs: ClassVar[dict] = {
        "size": 10,
        "subdivisions": 4,
    }
    physics_constructor: ClassVar[PhysicsConstructor] = PhysicsConstructor(
        YolkClothConstructor,
        CollisionConstructor,
    )
    color: ClassVar[tuple[float, float, float]] = (0.64, 0.64, 0.64)
