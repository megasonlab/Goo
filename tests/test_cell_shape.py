import pytest
import bpy
from goo import Cell, CellType
from goo.force import *
from goo.division import BisectDivisionLogic, BooleanDivisionLogic


@pytest.fixture
def setup_blender():
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset to empty scene
    cellsA = CellType("A", pattern="simple")
    cellsA.homo_adhesion_strength = 1000

    oriented_cell1 = cellsA.create_cell(
        name="A1", loc=(10, 0, 0), color=(0.5, 0.5, 0), scale=(2, 1.5, 1.5)
    )
    oriented_cell2 = cellsA.create_cell(
        name="A2", loc=(-10, 0, 0), color=(0.5, 0.5, 0), scale=(1.5, 2, 1.5)
    )
    oriented_cell3 = cellsA.create_cell(
        name="A3", loc=(0, 0, 0), color=(0.5, 0.5, 0), scale=(1.5, 1.5, 2)
    )
    oriented_cell4 = cellsA.create_cell(
        name="A4",
        loc=(0, 10, 0),
        color=(0.5, 0.5, 0),
        scale=(1, 1, 1),
        rotation=(45, 0, 0),
    )
    oriented_cell5 = cellsA.create_cell(
        name="A5",
        loc=(0, -10, 0),
        color=(0.5, 0.5, 0),
        scale=(1, 1, 1),
        rotation=(0, 45, 0),
    )
    sized_cell1 = cellsA.create_cell(
        name="A6", loc=(0, 0, 10), color=(0.5, 0.5, 0), scale=(1, 1, 1), size=1
    )
    sized_cell2 = cellsA.create_cell(
        name="A7", loc=(0, 0, 20), color=(0.5, 0.5, 0), size=2, scale=(2, 1, 1)
    )
    sized_cell3 = cellsA.create_cell(
        name="A8", loc=(0, 0, 30), color=(0.5, 0.5, 0), size=3, scale=(1, 2, 1)
    )
    sized_cell4 = cellsA.create_cell(
        name="A9", loc=(0, 0, 40), color=(0.5, 0.5, 0), size=4, scale=(1, 1, 1.33)
    )

    yield (
        oriented_cell1,
        oriented_cell2,
        oriented_cell3,
        oriented_cell4,
        oriented_cell5,
        sized_cell1,
        sized_cell2,
        sized_cell3,
        sized_cell4,
    )


def test_cell_major_axis(setup_blender):
    cell1, cell2, cell3 = setup_blender[0:3]

    major1 = cell1.major_axis()
    major2 = cell2.major_axis()
    major3 = cell3.major_axis()

    assert pytest.approx(major1._axis, abs=1e-6) == (-1, 0, 0) or (1, 0, 0)
    assert pytest.approx(major2._axis, abs=1e-6) == (0, -1, 0) or (0, 1, 0)
    assert pytest.approx(major3._axis, abs=1e-6) == (0, 0, -1) or (0, 0, 1)


def test_cell_major_cell_rotation(setup_blender):
    cell4, cell5 = setup_blender[3:5]
    major4 = cell4.major_axis()
    major5 = cell5.major_axis()

    assert pytest.approx(major4._axis, abs=1e-6) == (
        0.7071067811865476,
        0,
        0.7071067811865476,
    ) or (-0.7071067811865476, 0, -0.7071067811865476)
    assert pytest.approx(major5._axis, abs=1e-6) == (
        0,
        0.7071067811865476,
        0.7071067811865476,
    ) or (0, -0.7071067811865476, -0.7071067811865476)


def test_cell_area(setup_blender):
    cell1, cell2, cell3, cell4 = setup_blender[5:9]
    area1 = cell1.area()
    area2 = cell2.area()
    area3 = cell3.area()
    area4 = cell4.area()

    assert pytest.approx(area1, rel=1e-1) == 11.666
    assert pytest.approx(area2, rel=1e-1) == 46.664
    assert pytest.approx(area3, rel=1e-1) == 104.993
    assert pytest.approx(area4, rel=1e-1) == 186.655


def test_cell_aspect_ratio(setup_blender):
    cell1, cell2, cell3, cell4 = setup_blender[5:9]

    cell1.remesh()  # remesh because the cell is small
    # cell2.remesh()
    # cell3.remesh()
    # cell4.remesh()

    aspect1 = cell1.aspect_ratio()
    aspect2 = cell2.aspect_ratio()
    aspect3 = cell3.aspect_ratio()
    aspect4 = cell4.aspect_ratio()

    assert pytest.approx(aspect1, abs=1e-1) == 1
    assert pytest.approx(aspect2, abs=1e-1) == 2
    assert pytest.approx(aspect3, abs=1e-1) == 2
    assert pytest.approx(aspect4, abs=1e-1) == 1.33


def test_cell_sphericity(setup_blender):
    cell1, cell2, cell3, cell4 = setup_blender[5:9]

    cell1.remesh()  # remesh because the cell is small

    sphericity1 = cell1.sphericity()
    sphericity2 = cell2.sphericity()
    sphericity3 = cell3.sphericity()
    sphericity4 = cell4.sphericity()

    assert pytest.approx(sphericity1, abs=1e-1) == 1
    assert pytest.approx(sphericity2, abs=1e-1) == 1.5
    assert pytest.approx(sphericity3, abs=1e-1) == 1.5
    assert pytest.approx(sphericity4, abs=1e-1) == 1.2


# def test_sav_ratio(setup_blender):
#     cell1, cell2, cell3, cell4 = setup_blender[5:9]

#     cell1.remesh()  # remesh because the cell is small

#     sav1 = cell1.sav_ratio()
#     sav2 = cell2.sav_ratio()
#     sav3 = cell3.sav_ratio()
#     sav4 = cell4.sav_ratio()

#     assert pytest.approx(sav1, abs=1e-1) == 3.461
#     assert pytest.approx(sav2, abs=1e-1) == (0.804 or 0.530)
#     assert pytest.approx(sav3, abs=1e-1) == (0.804 or 0.530)
#     assert pytest.approx(sav4, abs=1e-1) == 0.83

# def test_cell_compactness(setup_blender):
#     cell1, cell2, cell3, cell4 = setup_blender[5:9]

#     cell1.remesh()  # remesh because the cell is small

#     compactness1 = cell1.compactness()
#     compactness2 = cell2.compactness()
#     compactness3 = cell3.compactness()
#     compactness4 = cell4.compactness()

#     assert pytest.approx(compactness1, abs=1e-1) == 0
#     assert pytest.approx(compactness2, abs=1e-1) == 0
#     assert pytest.approx(compactness3, abs=1e-1) == 0
#     assert pytest.approx(compactness4, abs=1e-1) == 0
