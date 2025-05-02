import bpy

from mathutils import Vector
from typing_extensions import override

from goo.utils import *


class Force(BlenderObject):
    """A force.

    Forces are represented in Blender by force field objects. They interact
    with cells to influence their motion.

    Args:
        obj: Blender object to be used as a representation for the force.

    Attributes:
        type (str): Type of force.
    """

    def __init__(self, obj: bpy.types.Object, type="FORCE"):
        super().__init__(obj)
        self.type = type

        # Instantiate force field object
        if not self.obj.field:
            with bpy.context.temp_override(active_object=self.obj, object=self.obj):
                bpy.ops.object.forcefield_toggle()
        self.enable()

    def enable(self):
        """Enables the force."""
        self.obj.field.type = self.type

    def disable(self):
        """Disables the force."""
        if self.obj.field:
            self.obj.field.type = "NONE"

    def enabled(self) -> bool:
        """Checks if the force field is enabled."""
        return self.obj.field and self.obj.field.type == self.type

    @property
    def strength(self) -> int:
        """Strength of the force field."""
        return self.obj.field.strength

    @strength.setter
    def strength(self, strength: int):
        self.obj.field.strength = strength

    @property
    def falloff(self) -> float:
        """Falloff power of the force. Strength at distance :math:`r` is given by
        :math:`\\text{strength} / r ^ \\text{falloff}`."""
        return self.obj.field.falloff

    @falloff.setter
    def falloff(self, falloff):
        self.obj.field.falloff_power = falloff

    @property
    def min_dist(self) -> float:
        """Minimum distance an object must be from a force to be affected."""
        if self.obj.field.use_min_distance:
            return self.obj.field.min_dist
        return None

    @min_dist.setter
    def min_dist(self, min_dist: float | None):
        if min_dist is None:
            self.obj.field.use_min_distance = False
        else:
            self.obj.field.use_min_distance = True
            self.obj.field.distance_min = min_dist

    @property
    def max_dist(self) -> float:
        """Maximum distance an object can be from a force to be affected."""
        if self.obj.field.use_max_distance:
            return self.obj.field.max_dist
        return None

    @max_dist.setter
    def max_dist(self, max_dist: float | None):
        if max_dist is None:
            self.obj.field.use_max_distance = False
        else:
            self.obj.field.use_max_distance = True
            self.obj.field.distance_max = max_dist

    @property
    def shape(self) -> int:
        """Shape of the force field."""
        return self.obj.field.shape

    @shape.setter
    def shape(self, shape: int):
        self.obj.field.shape = shape

    @property
    def impulse_clamp(self) -> int:
        """Impulse clamp of the force field."""
        return self.obj.modifiers["Cloth"].collision_settings.impulse_clamp

    @impulse_clamp.setter
    def impulse_clamp(self, impulse_clamp: int):
        self.obj.modifiers["Cloth"].collision_settings.impulse_clamp = impulse_clamp
        self.obj.modifiers["Cloth"].collision_settings.self_impulse_clamp = (
            impulse_clamp
        )


class AdhesionForce(Force):
    """An adhesion force."""

    def __init__(self, obj):
        super().__init__(obj, "FORCE")

    @override
    @property
    def strength(self) -> int:
        return -self.obj.field.strength

    @strength.setter
    def strength(self, strength: int):
        self.obj.field.strength = -strength


class MotionForce(Force):
    """A motion force."""

    def __init__(self, obj: bpy.types.Object):
        super().__init__(obj, "FORCE")
        obj.field.shape = "PLANE"
        obj.field.apply_to_rotation = False
        self.init_strength = None

    @override
    @property
    def strength(self):
        return -self.obj.field.strength

    @strength.setter
    def strength(self, strength):
        self.obj.field.strength = -strength

    # TODO: split into just change location and set rotation. "point_towards"
    def point_towards(self, loc: Vector):
        """Set location of a motion force, towards which a cell will move.

        Args:
            new_loc: Location of the motion force.
            target_loc: Location of the cell upon which the motion acts.
        """
        dir = loc - self.loc
        self.obj.rotation_euler = dir.to_track_quat("Z", "X").to_euler()


# TODO: remove because not used
def create_force(
    name: str,
    loc: tuple,
    strength: int,
    type: str = "FORCE",
    falloff: float = 0,
    min_dist: float | None = None,
    max_dist: float | None = None,
    shape: str = "POINT",
) -> Force:
    """Creates a new force field.

    Args:
        name: The name of the force field object.
        loc: The location of the force field object.
        strength: The strength of the force field.
        type: The type of the force field.
        falloff: The falloff power of the force field.
        min_dist: The minimum distance for the force field.
        max_dist: The maximum distance for the force field.
        shape: The shape of the force field. Defaults to "POINT".

    Returns:
        Force: The created force field object.
    """
    obj = bpy.data.objects.new(name, None)
    obj.location = loc
    force = Force(obj, type)

    force.strength = strength
    force.falloff = falloff
    force.min_dist = min_dist
    force.max_dist = max_dist
    force.shape = shape

    ForceCollection.global_forces().add(force)
    return force


def create_adhesion(
    strength: int,
    obj: bpy.types.Object | None = None,
    name: str | None = None,
    loc: tuple | None = None,
    shape: str = "POINT",
) -> AdhesionForce:
    """Creates a new adhesion force.

    Adhesion forces can either be created from cells, in which they are
    homotypic forces, Or they can be created de novo, in which they are
    heterotypic adhesion forces meant to allow two different cell types to
    interact with each other.

    Args:
        strength: Strength of the adhesion force.
        obj: Cell to use as origin of the adhesion force.
            If None, a new object is created.
        name: Name of the adhesion force.
        loc: Initial location of the adhesion force.
        shape: Shape of the adhesion force.
    """
    if obj is None:
        if name is None or loc is None:
            raise ValueError(
                "If obj is not provided, then both name and \
                    loc parameters must be provided."
            )
        obj = bpy.data.objects.new(name, None)
        obj.location = loc
        obj.hide_set(False)
    adhesion_force = AdhesionForce(obj)

    adhesion_force.strength = strength
    adhesion_force.shape = shape
    adhesion_force.min_dist = 0.6
    adhesion_force.max_dist = 1.6
    adhesion_force.falloff = 1
    return adhesion_force


def create_motion(name: str, loc: tuple, strength: int) -> MotionForce:
    """Creates a new motion force.

    Args:
        name: Name of the motion force.
        loc: Initial location of the motion force.
        strength: Strength of the motion force.
    """
    obj = bpy.data.objects.new(name, None)
    obj.location = loc
    force = MotionForce(obj)
    force.init_strength = strength
    force.strength = strength
    # obj.hide_render = True
    obj.hide_viewport = True
    return force


class ForceCollection:
    """A class representing a collection of forces.

    A force collection is represented by Blender collections of Blender forces.
    """

    _global_forces = None

    def __init__(self, name: str):
        self.col = bpy.data.collections.new(name)
        self.children = []

    @property
    def name(self) -> str:
        """Name of the collection of forces."""
        return self.col.name

    @name.setter
    def name(self, n):
        self.col.name = n

    def add(self, force: Union[Force, "ForceCollection"]):
        """Add a Force or Force Collection to the collection."""
        self.children.append(force)
        if isinstance(force, Force):
            self.col.objects.link(force.obj)
        elif isinstance(force, ForceCollection):
            self.col.children.link(force.col)

    def remove(self, force: Union[Force, "ForceCollection"]):
        """Remove a Force or Force Collection from the collection."""
        self.children.remove(force)
        if isinstance(force, Force):
            self.col.objects.unlink(force.obj)
        elif isinstance(force, ForceCollection):
            self.col.children.unlink(force.col)

    @property
    def forces(self):
        """List of all forces contained in this collection and subcollections."""
        forces = []
        for child in self.children:
            if isinstance(child, Force):
                forces.append(child)
            elif isinstance(child, ForceCollection):
                forces.extend(child.forces)
        return forces

    def show(self):
        """Add the Force Collection to the scene, making it visible in Blender."""
        if self.col.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(self.col)

    def hide(self):
        """Remove the Force Collection from the scene, making it invisible in Blender."""
        if self.col.name in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(self.col)

    @staticmethod
    def global_forces() -> "ForceCollection":
        """Collection of forces that affects all cells.

        Returns:
            The collection containing global forces.
        """
        if ForceCollection._global_forces is None:
            ForceCollection._global_forces = ForceCollection("globals")
            ForceCollection._global_forces.show()
        return ForceCollection._global_forces


class Boundary(BlenderObject):
    """A boundary for cells."""

    def setup_physics(self):
        """Set up physics for the boundary."""
        BoundaryCollisionConstructor().construct(self.obj)


def create_boundary(loc: tuple, size: float, mesh: str = "icosphere"):
    """Create a boundary.

    Args:
        loc: Center of the boundary.
        size: Radius of the boundary.
        mesh: Shape fo the boundary.
    """

    obj = create_mesh("Boundary", loc, mesh=mesh, size=size)
    bpy.context.scene.collection.objects.link(obj)

    boundary = Boundary(obj)
    boundary.setup_physics()
    boundary.hide()
    return boundary
