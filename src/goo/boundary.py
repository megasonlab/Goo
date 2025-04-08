from typing import Optional
from typing_extensions import override
import bpy
import numpy as np
from goo.utils import *


class Boundary(BlenderObject):
    """A boundary for cells."""

    def __init__(self, obj: bpy.types.Object, mat=None):
        super(Boundary, self).__init__(obj)
        self._mat = mat
        self.obj.data.materials.append(mat)

    def setup_physics(self):
        """Set up physics for the boundary."""
        BoundaryCollisionConstructor().construct(self.obj)
    
    @property
    def size(self) -> int:
        """Size of the boundary."""
        return self.obj.size
    
    @size.setter
    def size(self, size: int):
        self.obj.size = size


def create_boundary(loc: tuple, size: float, mesh: str = "icosphere"):
    """Create a boundary.

    Args:
        loc: Center of the boundary.
        size: Radius of the boundary.
        mesh: Shape of the boundary.
    """
    if mesh not in ["icosphere", "cube"]:
        raise ValueError(f"Unsupported mesh type: {mesh}."
                         "Supported types are 'icosphere' and 'cube'.")

    obj = create_mesh("Boundary", loc, mesh=mesh, size=size, subdivisions=4)
    bpy.context.scene.collection.objects.link(obj)

    boundary = Boundary(obj)
    boundary.setup_physics()
    boundary.hide()
    return boundary