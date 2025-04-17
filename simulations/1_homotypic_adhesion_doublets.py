from importlib import reload
import goo

reload(goo)
goo.reset_modules()
goo.reset_scene()

cellsA = goo.CellType("A", target_volume=50, pattern="simple")
cellsB = goo.CellType("B", target_volume=50, pattern="simple")
cellsC = goo.CellType("C", target_volume=50, pattern="simple")

cellsA.homo_adhesion_strength = 0
cellsB.homo_adhesion_strength = 250
cellsC.homo_adhesion_strength = 500

cellsA.create_cell("A1", (-5, +1.75, 0), color=(1, 1, 0), size=1.6)
cellsA.create_cell("A2", (-5, -1.75, 0), color=(1, 1, 0), size=1.6)

cellsB.create_cell("B1", (0, +1.75, 0), color=(0, 1, 1), size=1.6)
cellsB.create_cell("B2", (0, -1.75, 0), color=(0, 1, 1), size=1.6)

cellsC.create_cell("C1", (5, +1.75, 0), color=(1, 0, 1), size=1.6)
cellsC.create_cell("C2", (5, -1.75, 0), color=(1, 0, 1), size=1.6)

sim = goo.Simulator([cellsA, cellsB, cellsC], time=300)
sim.setup_world(seed=2024)
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.RecenterHandler(),
        goo.DataExporter(
            # path="/Users/antoine/Harvard/MegasonLab/GPU_backup/AntoineRuzette/goo/data/docker_export_test/out.json",
            options=goo.DataFlag.MOTION_PATH,
        ),
    ]
)

# sim.run()
