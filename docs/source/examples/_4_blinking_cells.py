from importlib import reload
import goo
from goo.gene import *
from goo.division import *
from goo.handler import *

reload(goo)
goo.reset_modules()
goo.reset_scene()

# Defining genes
x = Gene("x")
y = Gene("y")
z = Gene("z")

celltype1 = goo.CellType("cellA", pattern="simple", target_volume=70)
celltype1.homo_adhesion_strength = 500
celltype1.motion_strength = 1000
cell1 = celltype1.create_cell(name="cell1", loc=(-10, 0, 0), color=(0, 0, 0))
cell1.stiffness = 5

celltype2 = goo.CellType("cellB", pattern="simple", target_volume=70)
celltype2.homo_adhesion_strength = 500
celltype2.motion_strength = 1000
cell2 = celltype2.create_cell(name="cell1", loc=(10, 0, 0), color=(0, 0, 0))
cell2.stiffness = 5

network1 = GeneRegulatoryNetwork()
network1.load_circuits(
    DegFirstOrder(x, 0.1),
    DegFirstOrder(y, 0.1),
    DegFirstOrder(z, 0.1),
    ProdRepression(y, x, kcat=0.4, n=3),
    ProdRepression(z, y, kcat=0.4, n=3),
    ProdRepression(x, z, kcat=0.4, n=3),
)
cell1.grn = network1
cell1.metabolites = {x: 2, y: 0.1, z: 0.1}

network2 = GeneRegulatoryNetwork()
network2.load_circuits(
    DegFirstOrder(x, 0.1),
    DegFirstOrder(y, 0.1),
    DegFirstOrder(z, 0.1),
    ProdRepression(y, x, kcat=0.55, n=3),
    ProdRepression(z, y, kcat=0.55, n=3),
    ProdRepression(x, z, kcat=0.55, n=3),
)
cell2.grn = network2
cell2.metabolites = {x: 2, y: 0.1, z: 0.1}

sim = goo.Simulator(celltypes=[celltype1, celltype2], time=200, physics_dt=1)
sim.setup_world(seed=2025)
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=60, sigma=2),
        goo.RecenterHandler(),
        goo.RandomMotionHandler(distribution=goo.ForceDist.GAUSSIAN),
        goo.NetworkHandler(),
        goo.ColorizeHandler(goo.Colorizer.GENE, x, range=(1, 2)),
    ]
)