import pytest
import bpy
import goo
from goo import Cell, CellType
from goo.force import *
from goo.handler import RecenterHandler, GrowthPIDHandler, RemeshHandler
from goo.simulator import Simulator
from mathutils import Vector


@pytest.fixture
def setup_blender():
    # Create a Blender object to pass to the Cell constructor
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset to empty scene
    cellsA = CellType("A")
    cellsB = CellType("B")
    cellsA.homo_adhesion_strength = 500
    low_res_cell = cellsA.create_cell(
        "A1", (10, 0, 0), color=(0.5, 0, 0), target_volume=50, size=1
    )
    high_res_cell = cellsB.create_cell(
        "B1", (-10, 0, 0), color=(0.5, 0, 0), target_volume=50, size=1
    )

    sim_low_res = Simulator([cellsA], time=50, physics_dt=1)
    # cannot use setup_world in testing because requires Blender in non-headless mode
    # sim.setup_world()
    sim_low_res.add_handlers(
        [
            GrowthPIDHandler(),
            RecenterHandler(),
            RemeshHandler(freq=1, voxel_size=0.5),
        ]
    )

    sim_high_res = Simulator([cellsB], time=50, physics_dt=1)
    # cannot use setup_world in testing because requires Blender in non-headless mode
    # sim.setup_world()
    sim_high_res.add_handlers(
        [
            GrowthPIDHandler(),
            RecenterHandler(),
            RemeshHandler(freq=5, voxel_size=0.25),
        ]
    )
    yield low_res_cell, high_res_cell


def test_initial_mesh_resolution(setup_blender):
    low_res_cell, high_res_cell = setup_blender

    low_res_vert_count = len(Cell.vertices(low_res_cell))
    high_res_vert_count = len(Cell.vertices(high_res_cell))

    assert low_res_vert_count == high_res_vert_count


def test_remesher_resolution(setup_blender):
    low_res_cell, high_res_cell = setup_blender

    for i in range(5):
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

    low_res_vert_count = len(Cell.vertices(low_res_cell))
    high_res_vert_count = len(Cell.vertices(high_res_cell))

    assert low_res_vert_count < high_res_vert_count


def test_remesher_frequency(setup_blender):
    low_res_cell, high_res_cell = setup_blender

    for i in range(5):
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

    high_res_vert_count_t0 = len(Cell.vertices(high_res_cell))
    bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
    high_res_vert_count_t1 = len(Cell.vertices(high_res_cell))

    for i in range(5):
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

    low_res_vert_count_t0 = len(Cell.vertices(low_res_cell))
    low_res_vert_count_t1 = len(Cell.vertices(low_res_cell))

    assert high_res_vert_count_t0 == high_res_vert_count_t1
    assert low_res_vert_count_t0 == low_res_vert_count_t1
    assert low_res_vert_count_t0 != high_res_vert_count_t1
