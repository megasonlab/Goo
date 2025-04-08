import pytest
import bpy
import goo
from goo import Cell, CellType
from goo.force import *
from goo.handler import (
    RecenterHandler,
    GrowthPIDHandler,
    RandomMotionHandler,
    ForceDist,
)
from goo.simulator import Simulator
from goo.cell import *
import bmesh
from mathutils import Vector


@pytest.fixture
def setup_blender():
    # Create a Blender object to pass to the Cell constructor
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset to empty scene
    cellsA = CellType("A")
    cellsA.homo_adhesion_strength = 100
    cell = cellsA.create_cell(
        "A1", (0, 0, 0), color=(0.5, 0, 0), target_volume=50, size=1
    )

    sim = Simulator([cellsA], time=10, physics_dt=1)
    # cannot use setup_world in testing because requires Blender in non-headless mode
    # sim.setup_world()
    sim.set_seed(1)
    sim.add_handlers(
        [
            GrowthPIDHandler(),
            RandomMotionHandler(ForceDist.UNIFORM, strength=10000),
            RecenterHandler(),
        ]
    )

    yield cell


def test_next_cell_position(setup_blender):
    cell = setup_blender
    current_loc = Cell.COM(cell)

    bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

    next_random_loc = Cell.COM(cell)

    assert current_loc != next_random_loc


def test_random_motion_force_position(setup_blender):
    cell = setup_blender
    force_loc_t = Vector(cell.motion_force.loc)
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
    force_loc_t1 = Vector(cell.motion_force.loc)
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
    force_loc_t2 = Vector(cell.motion_force.loc)
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
    force_loc_t3 = Vector(cell.motion_force.loc)

    assert force_loc_t != force_loc_t1
    assert force_loc_t1 != force_loc_t2
    assert force_loc_t2 != force_loc_t3
    assert force_loc_t != force_loc_t1 != force_loc_t2 != force_loc_t3
