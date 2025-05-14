import goo
from importlib import reload

reload(goo)
goo.reset_modules()
goo.reset_scene()

cellsA = goo.CellType("A", target_volume=25, pattern="simple")
cellsB = goo.CellType("B", target_volume=25, pattern="simple")
cellsC = goo.CellType("C", target_volume=25, pattern="simple")

cellsA.homo_adhesion_strength = 0
cellsB.homo_adhesion_strength = 250
cellsC.homo_adhesion_strength = 500

cellsA.create_cell("A1", (+1.75, -5, 0), color=(0.5, 0, 0), size=1.6)
cellsA.create_cell("A2", (-1.75, -5, 0), color=(0.5, 0, 0), size=1.6)

cellsB.create_cell("B1", (+1.75, 0, 0), color=(0, 0.5, 0), size=1.6)
cellsB.create_cell("B2", (-1.75, 0, 0), color=(0, 0.5, 0), size=1.6)

cellsC.create_cell("C1", (+1.75, 5, 0), color=(0, 0, 0.5), size=1.6)
cellsC.create_cell("C2", (-1.75, 5, 0), color=(0, 0, 0.5), size=1.6)

sim = goo.Simulator([cellsA, cellsB, cellsC], time=300, physics_dt=1)
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.RecenterHandler(),
        goo.RandomMotionHandler(goo.ForceDist.GAUSSIAN, strength=750),
    ]
)
