import goo

goo.reset_modules()
goo.reset_scene()

celltype = goo.CellType("cellA", target_volume=70, pattern="simple")
celltype.homo_adhesion_strength = 1000
cell = celltype.create_cell(name="cell", loc=(0, 0, 0), color=(0, 1, 1))
cell.stiffness = 2

sim = goo.Simulator(celltypes=[celltype], time=500, physics_dt=1)
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.TimeDivisionHandler(goo.BisectDivisionLogic, mu=100, sigma=5),
        goo.RecenterHandler(),
        goo.RemeshHandler(),
    ]
)
