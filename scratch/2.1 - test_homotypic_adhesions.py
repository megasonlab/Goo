from importlib import reload
import goo
from goo.handler import *

reload(goo)
goo.reset_modules()
goo.reset_scene()

celltype = goo.CellType("cellsA", pattern="simple", homo_adhesion_strength=100)

locs = [
    (-1.36, -1, 0.2),
    (-0.2, -1, -1.36),
    (1.36, -1, -0.2),
    (0.2, -1, 1.36),
    (-0.73, 1, 1.1),
    (1.2, 1, 0.84),
    (0.92, 1, -1.08),
    (-1, 1, -0.82),
]

for i, loc in enumerate(locs):
    cell = celltype.create_cell(f"cell{i}", loc, size=1)
    cell.stiffness = 15
    cell.pressure = 1

sim = goo.Simulator([celltype])
sim.setup_world()
sim.add_handler(RecenterHandler())
