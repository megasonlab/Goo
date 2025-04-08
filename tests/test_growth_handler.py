import pytest
import bpy
import goo
from goo import Cell, CellType
from goo.force import *
from goo.handler import RecenterHandler, GrowthPIDHandler
from goo.simulator import Simulator
import bmesh
from mathutils import Vector


@pytest.fixture
def setup_blender():
    # Create a Blender object to pass to the Cell constructor
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset to empty scene
    cellsA = CellType("A")
    cellsB = CellType("B")
    cellsA.homo_adhesion_strength = 500
    small_cell = cellsA.create_cell(
        "A1", (10, 0, 0), color=(0.5, 0, 0), size=1, target_volume=50
    )
    large_cell = cellsB.create_cell(
        "A2", (-10, 0, 0), color=(0.5, 0, 0), size=1, target_volume=100
    )

    sim_small = Simulator([cellsA], time=105, physics_dt=1)
    # cannot use setup_world in testing because requires Blender in non-headless mode
    # sim.setup_world()
    sim_small.add_handlers(
        [
            GrowthPIDHandler(),
            RecenterHandler(),
        ]
    )

    sim_large = Simulator([cellsB], time=105, physics_dt=1)
    # cannot use setup_world in testing because requires Blender in non-headless mode
    # sim.setup_world()
    sim_large.add_handlers(
        [
            GrowthPIDHandler(),
            RecenterHandler(),
        ]
    )
    yield small_cell, large_cell


def test_growth_PID_handler(setup_blender: Cell):
    small_cell, large_cell = setup_blender

    for i in range(49):
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

    small_volume = small_cell.volume()
    large_volume = large_cell.volume()

    # same volume until reaching target volume
    assert pytest.approx(small_volume, rel=5e-1) == large_volume

    for i in range(49):
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

    small_volume = small_cell.volume()
    large_volume = large_cell.volume()

    assert small_volume < large_volume
    assert large_volume == pytest.approx(small_volume * 2, rel=1e-1)
