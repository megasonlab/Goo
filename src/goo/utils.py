from functools import reduce
from typing import Union

import bpy
import bmesh
from bpy.types import Modifier, ClothModifier
from mathutils import *


class BlenderObject:
    def __init__(self, obj: bpy.types.Object):
        self.obj = obj

    @property
    def name(self):
        return self.obj.name

    @name.setter
    def name(self, name):
        self.obj.name = name

    @property
    def loc(self):
        return self.obj.location

    @loc.setter
    def loc(self, loc):
        self.obj.location = loc

    def hide(self):
        self.obj.hide_set(True)

    # ----- CUSTOM PROPERTIES -----
    def __setitem__(self, k: str, v: Union[float, list[float], int, list[int], str]):
        self.obj[k] = v

    def __contains__(self, k: str):
        return k in self.obj.keys()

    def __getitem__(self, k):
        return self.obj[k]


class Axis:
    def __init__(self, axis, start, end, world_matrix):
        """
        :param axis: Vector of axis
        :param start: Vector of start endpoint in object space
        :param end: Vector of end endpoint in object space
        :param world_matrix: 4x4 matrix of object to world transformation
        :param local_coords: if coordinates given are in local space
        """
        self._axis = axis
        self._start = start
        self._end = end
        self._matrix_world = world_matrix

    def axis(self, local_coords=False):
        if local_coords:
            axis = self._axis.copy()
            axis.rotate(self._matrix_world.to_quaternion().inverted())
            return axis
        return self._axis

    def endpoints(self, local_coords=False):
        mat = self._matrix_world.inverted() if local_coords else Matrix.Identity(4)
        return [mat @ self._start, mat @ self._end]

    def length(self, local_coords=False):
        start, end = self.endpoints(local_coords)
        return (end - start).length


# ----- BLENDER FUNCTIONS -----
def create_mesh(
    name,
    loc,
    mesh="icosphere",
    size=1.5,
    rotation=(0, 0, 0),
    scale=(1, 1, 1),
    subdivisions=2,
    **kwargs,
):
    bm = bmesh.new()

    if isinstance(rotation, tuple):
        rotation = Euler(rotation)
    elif isinstance(rotation, Quaternion):
        rotation = rotation.to_euler()

    match mesh:
        case "icosphere":
            bmesh.ops.create_icosphere(
                bm, subdivisions=subdivisions, radius=size, **kwargs
            )
            # bmesh.ops.beautify_fill(bm, faces=bm.faces, edges=bm.edges)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
        case "plane":
            bmesh.ops.create_grid(
                bm, x_segments=1, y_segments=1, size=size / 2, **kwargs
            )
        case "monkey":
            bmesh.ops.create_monkey(bm, **kwargs)
        case "cube":
            bmesh.ops.create_cube(bm, size=size, **kwargs)
            # bmesh.ops.beautify_fill(bm, faces=bm.faces, edges=bm.edges)
            # bmesh.ops.triangulate(bm, faces=bm.faces[:])
        case _:
            raise ValueError(
                """mesh must be one of "icosphere", "plane", "monkey" or "cube"."""
            )

    me = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new(name, me)
    obj.location = loc
    obj.rotation_euler = rotation
    obj.scale = scale

    return obj


def create_material(name, color):
    # Create a new material
    mat = bpy.data.materials.new(name=name)
    r, g, b = color
    mat.diffuse_color = (r, g, b, 1.0)

    # Set material properties to create a matte finish
    mat.metallic = 1.0  # No metallic reflection
    mat.roughness = 1.0  # Maximum roughness for a matte finish
    mat.specular_intensity = 1.0  # No specular reflection
    mat.use_nodes = False  # Ensure nodes are not used

    return mat


# --- PHYSICS MODIFIER CONSTRUCTORS ---
class PhysicsConstructor:
    def __init__(self, *mod_contructors: "ModConstructor"):
        self.mod_constructors = [*mod_contructors]

    def __call__(self, bobj: BlenderObject):
        for mod_constructor in self.mod_constructors:
            mod_constructor().construct(bobj.obj)


class ModConstructor:
    name = ""
    type = ""

    def construct(self, obj):
        mod = obj.modifiers.new(name=self.name, type=self.type)
        self.setup_mod(mod)

    def setup_mod(self, mod: Modifier):
        pass


class ClothConstructor(ModConstructor):
    name = "Cloth"
    type = "CLOTH"

    def setup_mod(self, mod: ClothModifier):
        stiffness = 1
        pressure = 0.01

        mod.settings.quality = 10
        mod.settings.air_damping = 10
        mod.settings.bending_model = "ANGULAR"
        mod.settings.mass = 1
        mod.settings.time_scale = 1
        mod.point_cache.frame_start = bpy.context.scene.frame_start
        mod.point_cache.frame_end = bpy.context.scene.frame_end
        # Cloth > Stiffness
        mod.settings.tension_stiffness = stiffness
        mod.settings.compression_stiffness = stiffness
        mod.settings.shear_stiffness = stiffness
        mod.settings.bending_stiffness = 1
        # Cloth > Damping
        mod.settings.tension_damping = 50
        mod.settings.compression_damping = 50
        mod.settings.shear_damping = 50
        mod.settings.bending_damping = 0.5
        # Cloth > Pressure
        mod.settings.use_pressure = True
        mod.settings.uniform_pressure_force = pressure
        mod.settings.use_pressure_volume = True
        mod.settings.target_volume = 1
        mod.settings.pressure_factor = 2
        # mod.settings.fluid_density = 1.05
        # Cloth > Collisions
        mod.collision_settings.collision_quality = 4
        mod.collision_settings.use_collision = True
        mod.collision_settings.use_self_collision = True
        mod.collision_settings.self_friction = 0
        mod.collision_settings.friction = 0
        mod.collision_settings.self_distance_min = 0.01
        mod.collision_settings.distance_min = 0.01
        mod.collision_settings.self_impulse_clamp = 100
        mod.collision_settings.impulse_clamp = 100

        # Cloth > Field Weights
        mod.settings.effector_weights.gravity = 0


class SimpleClothConstructor(ClothConstructor):
    def setup_mod(self, mod):
        super().setup_mod(mod)
        mod.settings.mass = 10


class YolkClothConstructor(ClothConstructor):
    def setup_mod(self, mod: ClothModifier):
        stiffness = 5
        pressure = 5.2

        mod.settings.quality = 10
        mod.settings.air_damping = 10
        mod.settings.bending_model = "ANGULAR"
        mod.settings.mass = 2
        mod.settings.time_scale = 1
        # Cloth > Stiffness
        mod.settings.tension_stiffness = stiffness
        mod.settings.compression_stiffness = stiffness
        mod.settings.shear_stiffness = stiffness
        mod.settings.bending_stiffness = stiffness
        # Cloth > Damping
        mod.settings.tension_damping = 50
        mod.settings.compression_damping = 50
        mod.settings.shear_damping = 50
        mod.settings.bending_damping = 0.5
        # Cloth > Internal Springs
        mod.settings.use_internal_springs = True
        mod.settings.internal_spring_max_length = 1
        mod.settings.internal_spring_max_diversion = 0.785398
        mod.settings.internal_spring_normal_check = False
        mod.settings.internal_tension_stiffness = 10000
        mod.settings.internal_compression_stiffness = 10000
        mod.settings.internal_tension_stiffness_max = 10000
        mod.settings.internal_compression_stiffness_max = 10000
        # Cloth > Pressure
        mod.settings.use_pressure = True
        mod.settings.uniform_pressure_force = pressure
        mod.settings.use_pressure_volume = True
        mod.settings.target_volume = 1
        mod.settings.pressure_factor = 2
        mod.settings.fluid_density = 1.15
        # Cloth > Collisions
        mod.collision_settings.collision_quality = 10
        mod.collision_settings.use_collision = True
        mod.collision_settings.use_self_collision = True
        mod.collision_settings.self_friction = 0
        mod.collision_settings.friction = 0
        mod.collision_settings.self_distance_min = 0.005
        mod.collision_settings.distance_min = 0.005
        mod.collision_settings.self_impulse_clamp = 0


class CollisionConstructor(ModConstructor):
    name = "Collision"
    type = "COLLISION"

    def setup_mod(self, mod: bpy.types.CollisionModifier):
        mod.settings.use_culling = True
        mod.settings.damping = 1
        mod.settings.thickness_outer = 0.025
        mod.settings.thickness_inner = 0.25
        mod.settings.cloth_friction = 0
        mod.settings.use_normal = False


class BoundaryCollisionConstructor(CollisionConstructor):
    def setup_mod(self, mod: bpy.types.CollisionModifier):
        super(BoundaryCollisionConstructor, self).setup_mod(mod)
        mod.settings.use_culling = False


class SubsurfConstructor(ModConstructor):
    name = "Subdivision"
    type = "SUBSURF"

    def setup_mod(self, mod: bpy.types.SubsurfModifier):
        mod.subdivision_type = "CATMULL_CLARK"
        mod.levels = 1
        mod.render_levels = 1


# not stable
class RemeshConstructor(ModConstructor):
    name = "Remesh"
    type = "REMESH"

    def setup_mod(self, mod: bpy.types.RemeshModifier):
        mod.mode = "VOXEL"
        mod.voxel_size = 0.75
        mod.adaptivity = 0
        mod.use_remove_disconnected = True
        mod.use_smooth_shade = True
        mod.show_in_editmode = True
