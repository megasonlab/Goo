from importlib import reload
import goo

reload(goo)
goo.reset_modules()
goo.reset_scene()

# Defining cells
celltype = goo.CellType("cellA", target_volume=70, pattern="simple")
celltype.homo_adhesion_strength = 500

cell = celltype.create_cell(name="cell", loc=(0, 0, 0), color=(0, 0, 0))
cell.stiffness = 5
output_dir = "~/pointcloud"

sim = goo.Simulator(celltypes=[celltype], time=500, physics_dt=1)
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=60, sigma=2),
        goo.RecenterHandler(),
        goo.RemeshHandler(),
        goo.SliceExporter(output_dir=output_dir, downscale=(8, 4, 4), microscope_dt=20),
    ]
)