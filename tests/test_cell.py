import pytest
import bpy
from goo import Cell, CellType
from goo.force import *
from goo.division import BisectDivisionLogic
from mathutils import Vector


@pytest.fixture
def setup_blender():
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset to empty scene
    cellsA = CellType("A", pattern="simple")
    cellsA.homo_adhesion_strength = 1000
    cell = cellsA.create_cell("A1", (0, 0, 0), color=(0.5, 0, 0), size=2)

    yield cell


def test_cell_creation():
    cellsA = CellType("A")
    cell = cellsA.create_cell("A2", (1, 1, 1), color=(1, 0, 0))
    assert cell.name == "A2"
    assert cell.loc == Vector((1, 1, 1))
    assert cell.color == (1, 0, 0)


def test_get_volume(setup_blender: Cell):
    cell = setup_blender
    volume = cell.volume()
    expected_volume = 29.27
    assert volume == pytest.approx(expected_volume, rel=1e-1)


def test_get_cell_name(setup_blender: Cell):
    cell = setup_blender
    name = cell.name
    expected_name = "A1"
    assert name == expected_name


def test_get_type_name(setup_blender: Cell):
    cell = setup_blender
    type_name = cell.celltype.name
    expected_name = "A"
    assert type_name == expected_name


def test_get_cell_number(setup_blender: Cell):
    cell = setup_blender
    type = cell.celltype
    cells_len = len(type.cells)
    assert cells_len == 1


def test_COM(setup_blender: Cell):
    cell = setup_blender
    COM = cell.COM()
    expected_COM = Vector((0, 0, 0))
    assert COM.x == pytest.approx(expected_COM.x, abs=1e-6)
    assert COM.y == pytest.approx(expected_COM.y, abs=1e-6)
    assert COM.z == pytest.approx(expected_COM.z, abs=1e-6)


def test_homotypic_adhesion_strength(setup_blender: Cell):
    cell = setup_blender
    strength = cell.celltype.homo_adhesion_strength
    expected_strength = 1000
    assert strength == expected_strength


def test_cell_stiffness(setup_blender: Cell):
    cell = setup_blender
    stiffness = cell.stiffness
    expected_stiffness = 1
    assert stiffness == expected_stiffness


def test_update_cell_stiffness(setup_blender: Cell):
    cell = setup_blender
    cell.stiffness = 10
    stiffness = cell.stiffness
    expected_stiffness = 10
    assert stiffness == expected_stiffness


def test_cell_pressure(setup_blender: Cell):
    cell = setup_blender
    pressure = cell.pressure
    expected_pressure = 0.01
    assert pressure == pytest.approx(expected_pressure, abs=1e-5)


def test_update_cell_pressure(setup_blender: Cell):
    cell = setup_blender
    cell.pressure = 10
    pressure = cell.pressure
    expected_pressure = 10
    assert pressure == expected_pressure


def test_motion_strength(setup_blender: Cell):
    cell = setup_blender
    strength = cell.celltype.motion_strength
    expected_strength = 0
    assert strength == expected_strength


def test_updated_motion_strength(setup_blender: Cell):
    cell = setup_blender
    cell.celltype.motion_strength = 1000
    strength = cell.celltype.motion_strength
    expected_strength = 1000
    assert strength == expected_strength


def test_cell_material(setup_blender: Cell):
    cell = setup_blender
    mat = cell.color
    expected_mat = (0.5, 0, 0)
    assert mat == expected_mat


def test_update_cell_material(setup_blender: Cell):
    cell = setup_blender
    cell.recolor((0.5, 0.5, 0.5))
    mat = cell.color
    expected_mat = (0.5, 0.5, 0.5)
    assert mat == expected_mat


def test_disable_enable_physics(setup_blender: Cell):
    cell = setup_blender
    cell.disable_physics()
    assert not cell.physics_enabled
    cell.enable_physics()
    assert cell.physics_enabled


def test_recenter(setup_blender: Cell):
    cell = setup_blender
    initial_loc = cell.loc
    cell.recenter()
    # has not moved
    assert cell.loc == initial_loc


def test_remesh(setup_blender: Cell):
    cell = setup_blender
    initial_vert_count = len(cell.vertices())
    cell.remesh(voxel_size=0.1)  # Use a smaller voxel size
    new_vert_count = len(cell.vertices())
    assert new_vert_count != initial_vert_count
