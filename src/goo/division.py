from typing import Callable

import bpy
import bmesh
from mathutils import Vector
import numpy as np

from goo.cell import Cell
from goo.utils import *
from goo.handler import Handler
from goo.molecule import DiffusionSystem

from typing_extensions import override


class DivisionLogic:
    """Base class for defining division logic for cells."""

    def make_divide(self, mother: Cell) -> tuple[Cell, Cell]:
        """Divide a mother cell into two daughter cells.

        Args:
            mother: The mother cell to divide.

        Returns:
            A tuple containing the two daughter cells that will result from the
            division.

        Note:
            The meshes of the daughter should not be updated at this stage.
            `flush()` must be called to update all cells at once so that they do
            not interfere with each other.
        """
        pass

    def flush(self):
        """Finish performing all stored divisions."""
        pass


class BisectDivisionLogic(DivisionLogic):
    """Division logic that uses low-level `bmesh` operations, i.e.
    `bmesh.ops.bisect_plane` to divide a cell along its major axis.

    Attributes:
        margin (float): Distance of margin between divided cells.
    """

    def __init__(self, margin=0.025):
        self.margin = margin
        self.to_flush = []

    @override
    def make_divide(self, mother):
        com = mother.COM(local_coords=True)
        axis = mother.major_axis().axis(local_coords=True)
        base_name = mother.name

        m_mb = self._bisect(mother.obj_eval, com, axis, True, self.margin)
        d_mb = self._bisect(mother.obj_eval, com, axis, False, self.margin)

        daughter = mother.copy()

        mother.name = base_name + ".0"
        daughter.name = base_name + ".1"

        self.to_flush.append((m_mb, mother))
        self.to_flush.append((d_mb, daughter))

        return mother, daughter

    def _bisect(
        self,
        obj_eval: bpy.types.Object,
        com: Vector,
        axis: Axis,
        inner: bool,
        margin: float,
    ):
        """Bisect a mesh along a plane defined by center of mass and axis.

        Args:
            obj_eval: The evaluated object.
            com: The center of mass of the mesh.
            axis: The major axis of the mesh.
            inner: Whether to clear inner or outer part of the bisection.
            margin: The margin used for bisection.

        Returns:
            The `bmesh` object containing the resulting bisection.
        """
        bm = bmesh.new()
        bm.from_mesh(obj_eval.to_mesh())

        # bisect with plane
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        plane_co = com + axis * margin / 2 if inner else com - axis * margin / 2

        result = bmesh.ops.bisect_plane(
            bm,
            geom=geom,
            plane_co=plane_co,
            plane_no=axis,
            clear_inner=inner,
            clear_outer=not inner,
        )

        # fill in bisected face
        edges = [e for e in result["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        bmesh.ops.edgeloop_fill(bm, edges=edges)

        return bm

    @override
    def flush(self):
        for bm, cell in self.to_flush:
            bm.to_mesh(cell.obj.data)
            bm.free()
            cell.remesh()
            cell.recenter()
        self.to_flush.clear()


class BooleanDivisionLogic(DivisionLogic):
    """Division logic that creates a plane of division and applies the Boolean
    modifier to create a division."""

    # TODO: Update to work with physics.
    def __init__(self):
        pass

    @override
    def make_divide(self, mother: Cell):

        plane = self._create_division_plane(
            mother.name, mother.major_axis(), mother.COM()
        )
        obj = mother.obj

        # cut mother cell by division plane
        bpy.context.view_layer.objects.active = obj
        bool_mod = obj.modifiers.new(name="Boolean", type="BOOLEAN")
        bool_mod.operand_type = "OBJECT"
        bool_mod.object = plane
        bool_mod.operation = "DIFFERENCE"
        bool_mod.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)

        # separate two daughter cells
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")

        daughter = Cell(bpy.context.selected_objects[0])
        daughter.obj.select_set(False)

        daughter.name = mother.name + ".1"
        mother.name = mother.name + ".0"

        # remesh daughter cells
        mother.remesh()
        daughter.remesh()

        # clean up
        bpy.data.meshes.remove(plane.data, do_unlink=True)

        return mother, daughter

    @override
    def flush(self):
        pass

    def _create_division_plane(self, name, major_axis, com, collection=None):
        """
        Creates a plane orthogonal to the long axis vector
        and passing through the cell's center of mass.
        """
        # Define new plane
        plane = create_mesh(
            f"{name}_division_plane",
            loc=com,
            mesh="plane",
            size=major_axis.length() + 1,
            rotation=major_axis.axis().to_track_quat("Z", "Y"),
        )
        bpy.context.scene.collection.objects.link(plane)

        # Add thickness to plane
        solid_mod = plane.modifiers.new(name="Solidify", type="SOLIDIFY")
        solid_mod.offset = 0
        solid_mod.thickness = 0.025

        plane.hide_set(True)
        return plane


class DivisionHandler(Handler):
    """Handler for managing cell division processes.

    This handler is responsible for managing the division of cells based on the
    provided division logic. It determines which cells are eligible for division
    and performs the division process.

    Attributes:
        division_logic (DivisionLogic): The division logic used to execute cell
            division.
    """

    def __init__(self, division_logic, mu, sigma):
        self.division_logic = division_logic()
        self.mu = mu
        self.sigma = sigma

    @override
    def setup(
        self,
        get_cells: Callable[[], list[Cell]],
        get_diffsystems: Callable[[], list[DiffusionSystem]],
        dt: float,
    ):
        super(DivisionHandler, self).setup(get_cells, get_diffsystems, dt)
        for cell in self.get_cells():
            cell["divided"] = False
        self._cells_to_update = []

    def can_divide(self, cell: Cell) -> bool:
        """Check if a cell is eligible for division.

        This method must be implemented by all subclasses.

        Args:
            cell: The cell to check.

        Returns:
            True if the cell can divide, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement can_divide() method.")

    def update_on_divide(self, cell: Cell):
        """Perform updates after a cell has divided.

        This method can be overridden by subclasses to perform additional updates
        (e.g. set a property) after a cell has divided.

        Args:
            cell: The cell that has divided.
        """
        pass

    @override
    def run(self, scene, depsgraph):
        for cell in self._cells_to_update:
            cell.enable_physics()
            cell.cloth_mod.point_cache.frame_start = scene.frame_current
            cell["divided"] = False
        self._cells_to_update.clear()

        for cell in self.get_cells():
            if self.can_divide(cell):
                mother, daughter = cell.divide(self.division_logic)
                self.update_on_divide(mother)
                self.update_on_divide(daughter)

                if mother.physics_enabled:
                    self._cells_to_update.extend([mother, daughter])

        for cell in self._cells_to_update:
            cell.disable_physics()
            cell["divided"] = True
        self.division_logic.flush()


class TimeDivisionHandler(DivisionHandler):
    """Division handler that determines eligibility based on
    time from last divsion.

    Attributes:
        division_logic (DivisionLogic): see base class.
        mu (float): Time interval between cell divisions.
        var (float): Variance in the time interval.
    """

    def __init__(self, division_logic, mu=20, sigma=0):
        super(TimeDivisionHandler, self).__init__(division_logic, mu, sigma)

    @override
    def setup(self, get_cells, get_diffsystem, dt):
        super(TimeDivisionHandler, self).setup(get_cells, get_diffsystem, dt)
        for cell in self.get_cells():
            cell["last_division_time"] = 0

    @override
    def can_divide(self, cell: Cell):
        time = bpy.context.scene.frame_current * self.dt
        if "last_division_time" not in cell:
            cell["last_division_time"] = time
            return False
        # implement variance too
        div_time = int(np.random.normal(self.mu, self.sigma))
        return time - cell["last_division_time"] >= div_time

    @override
    def update_on_divide(self, cell: Cell):
        time = bpy.context.scene.frame_current * self.dt
        cell["last_division_time"] = time


class SizeDivisionHandler(DivisionHandler):
    """Division handler that determines eligibility based on
    size of cell.

    Attributes:
        division_logic (DivisionLogic): see base class.
        threshold (float): minimum size of cell able to divide.
    """

    def __init__(self, division_logic, mu=30, sigma=0):
        super(SizeDivisionHandler, self).__init__(division_logic, mu, sigma)

    @override
    def can_divide(self, cell: Cell):
        # time = bpy.context.scene.frame_current * self.dt
        # if "last_division_time" not in cell:
        #     cell["last_division_time"] = time
            
        div_volume = np.random.normal(self.mu, self.sigma)
        return cell.volume() >= div_volume
    
    @override
    def update_on_divide(self, cell: Cell):
        time = bpy.context.scene.frame_current * self.dt
        cell["last_division_time"] = time