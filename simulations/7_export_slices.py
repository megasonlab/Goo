from importlib import reload
import goo

reload(goo)
goo.reset_modules()
goo.reset_scene()

# Defining cells
celltype = goo.CellType("cellA", target_volume=70, pattern="simple")
celltype.homo_adhesion_strength = 500

cell = celltype.create_cell(name="cell", loc=(1, 1, 0), color=(0, 0, 0))
cell1 = celltype.create_cell(name="cell", loc=(4, 1, -0.5), color=(0, 0, 0))
cell2 = celltype.create_cell(name="cell", loc=(-1, 4, 0.4), color=(0, 0, 0))
cell3 = celltype.create_cell(name="cell", loc=(0.3, -2, -0.1), color=(0, 0, 0))
cell4 = celltype.create_cell(name="cell", loc=(3.6, -2.5, 0), color=(0, 0, 0))
cell5 = celltype.create_cell(name="cell", loc=(-1.8, 0.6, -0.5), color=(0, 0, 0))
cell.stiffness = 5
cell1.stiffness = 5
cell2.stiffness = 5
cell3.stiffness = 5
cell4.stiffness = 5
cell5.stiffness = 5
output_dir = "/Users/antoine/Harvard/MegasonLab/GPU_backup/AntoineRuzette/goo/data/slice_exporter/20250320_microsim/pointcloud_highres"

sim = goo.Simulator(celltypes=[celltype], time=500, physics_dt=1)
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=60, sigma=2),
        goo.RecenterHandler(),
        goo.RemeshHandler(),
        goo.SliceExporter(output_dir=output_dir, downscale=(8, 4, 4)),
    ]
)