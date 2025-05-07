import goo

goo.reset_modules()
goo.reset_scene()

# Defining genes
x = goo.Gene("x")
y = goo.Gene("y")
z = goo.Gene("z")

celltype = goo.CellType("cellA", pattern="simple", target_volume=125)
celltype.homo_adhesion_strength = 500
celltype.motion_strength = 1000
cell = celltype.create_cell(name="cell1", loc=(0, 0, 0), color=(0, 0, 0))
cell.stiffness = 5

network1 = goo.GeneRegulatoryNetwork()
network1.load_circuits(
    goo.DegFirstOrder(x, 0.04),
    goo.DegFirstOrder(y, 0.04),
    goo.DegFirstOrder(z, 0.04),
    goo.ProdRepression(y, x, kcat=0.5, n=3),
    goo.ProdRepression(z, y, kcat=0.5, n=3),
    goo.ProdRepression(x, z, kcat=0.5, n=3)
    )
cell.grn = network1
cell.gene_concs = {x: 2, y: 0.5, z: 0.5}
cell.link_gene_to_property(gene=x, property="motion_strength")

sim = goo.Simulator(celltypes=[celltype], time=1000, physics_dt=1)
sim.setup_world(seed=2025)
sim.add_handlers(
    [
        goo.GrowthPIDHandler(),
        goo.SizeDivisionHandler(goo.BisectDivisionLogic, mu=125, sigma=3),
        goo.RecenterHandler(),
        goo.NetworkHandler(),
        goo.ColorizeHandler(goo.Colorizer.GENE, x, range=(1, 2)),
        goo.RandomMotionHandler(goo.ForceDist.GAUSSIAN)
    ]
)