# import goo

# goo.reset_modules()
# goo.reset_scene()

# # Defining cells
# x = goo.Gene("x")
# y = goo.Gene("y")
# z = goo.Gene("z")

# celltype = goo.CellType("cellA", pattern="simple", target_volume=70)
# celltype.homo_adhesion_strength = 500
# cell = celltype.create_cell(name="cell1", loc=(0, 0, 0), color=(0, 0, 0))
# cell.stiffness = 5

# network1 = goo.GeneRegulatoryNetwork()
# network1.load_circuits(
#     goo.DegFirstOrder(x, 0.1),
#     goo.DegFirstOrder(y, 0.1),
#     goo.DegFirstOrder(z, 0.1),
#     goo.ProdRepression(y, x, kcat=0.4, n=3),
#     goo.ProdRepression(z, y, kcat=0.4, n=3),
#     goo.ProdRepression(x, z, kcat=0.4, n=3),
# )
# cell.grn = network1
# cell.gene_concs = {x: 2, y: 0.1, z: 0.1}

# sim = goo.Simulator(celltypes=[celltype], time=200, physics_dt=1)
# sim.setup_world(seed=2025)
# sim.add_handlers(
#     [
#         goo.GrowthPIDHandler(),
#         goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=60, sigma=2),
#         goo.RecenterHandler(),
#         goo.RemeshHandler(),
#         goo.NetworkHandler(),
#         goo.ColorizeHandler(goo.Colorizer.GENE, x, range=(1, 2)),
#                 goo.DataExporter(
#             path="/Users/antoine/Harvard/MegasonLab/GPU_backup/AntoineRuzette/goo/data/out"
#         ),
#     ]
# )

from importlib import reload

import goo


reload(goo)
goo.reset_modules()
goo.reset_scene()

# Defining cells
celltype = goo.CellType("cellA", target_volume=70, pattern="simple")
celltype.homo_adhesion_strength = 500
celltype.motion_strength = 100
cell = celltype.create_cell(name="cell", loc=(0, 0, 0), color=(0, 1, 1))
cell.stiffness = 5

mol = goo.Molecule("mol", conc=5, D=0, gradient="linear")
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
        goo.MolecularHandler(),
        goo.ConcentrationVisualizationHandler(spacing=1),
        goo.DataExporter(
            path="/Users/antoine/Harvard/MegasonLab/GPU_backup/AntoineRuzette/goo/data/out"
        ),
    ]
)
