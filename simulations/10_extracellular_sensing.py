from importlib import reload
import goo


reload(goo)
goo.reset_modules()
goo.reset_scene()

# Defining cells
celltype = goo.CellType("cellA", target_volume=70, pattern="simple")
celltype.homo_adhesion_strength = 500
cell = celltype.create_cell(name="cell", loc=(0, 0, 0), color=(0, 1, 1))
cell.stiffness = 5

mol = goo.Molecule("mol", conc=1, D=1, gradient="linear")
diffsys = goo.DiffusionSystem(molecules=[mol])
cell.diffsys = diffsys
cell.link_molecule_to_property(mol, "motion_direction")

sim = goo.Simulator(celltypes=[celltype], time=500, physics_dt=1, diffsystems=[diffsys])
sim.setup_world()
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
#        goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=60, sigma=2),
        goo.RecenterHandler(),
#        goo.RemeshHandler(),
        goo.ColorizeHandler(goo.Colorizer.RANDOM),
        goo.DiffusionHandler(),
#        goo.DataExporter("/Users/antoine/Harvard/MegasonLab/Goo-1/paper/data/test.h5" , goo.DataFlag.ALL),
    ]
)
