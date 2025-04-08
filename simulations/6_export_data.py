import goo

goo.reset_modules()
goo.reset_scene()

# Defining cells
celltype = goo.CellType("cellA", target_volume=50, pattern="simple")
celltype.homo_adhesion_strength = 1000
cell1 = celltype.create_cell(name="cell", loc=(0, 0, 0), color=(0, 1, 1))
cell1.stiffness = 2
cell1.pressure = 5

sim = goo.Simulator(celltypes=[celltype], time=500, physics_dt=1)
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.RecenterHandler(),
        goo.RandomMotionHandler(distribution=goo.ForceDist.UNIFORM, max_strength=250),
        goo.DataExporter(
            path="tmp/out.json",
            options=goo.DataFlag.CELL_CONCENTRATIONS
            | goo.DataFlag.MOTION_PATH,  # or ALL flag
        ),
    ]
)
