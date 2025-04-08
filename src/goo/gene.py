from dataclasses import dataclass
from typing_extensions import override

import antimony
import roadrunner
from mathutils import Vector

from .molecule import DiffusionSystem


class Gene:
    """Gene class to represent a gene in the gene regulatory network.
    
    Args:
        name: Name of the gene.
    """
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hash(self) == hash(other)


class Circuit:
    pass


@dataclass
class DegFirstOrder(Circuit):
    x: Gene
    k: float
    """d[x]/dt = -k[x]"""


@dataclass
class ProdAnd(Circuit):
    """d[z]/dt = k([x] AND [y]), optional substrate s consumed, 
    and optional leaky factor a0."""

    z: Gene
    x: Gene
    y: Gene
    k: float
    nx: float = 2
    ny: float = 2
    s: Gene = None
    a0: float = 0


@dataclass
class ProdActivation(Circuit):
    """d[y]/dt = kcat * [x]**n / (Km + [x]**n), optional substrate s consumed, 
    and optional leaky factor a0."""

    y: Gene
    x: Gene
    kcat: float
    Km: float = 1
    n: float = 2
    s: Gene = None
    a0: float = 0


@dataclass
class ProdRepression(Circuit):
    """d[y]/dt = kcat / (Km + [x]**n), optional substrate s consumed, 
    and optional leaky factor a0."""

    y: Gene
    x: Gene
    kcat: float
    Km: float = 1
    n: float = 2
    s: Gene = None
    a0: float = 0


class CircuitEngine:
    def __init__(self):
        pass

    def copy(self) -> "CircuitEngine":
        """Return copy of engine."""
        pass

    def load_circuits(self, *circuits: Circuit):
        """Load engine with model network."""
        pass

    def step(self, metabolite_concs: dict, dt: float):
        """Take 1 frame step of time dt to calculate new metabolite concentrations."""
        pass

    def retrieve_concs(self) -> dict[Gene, float]:
        """Get current metabolite concentrations."""
        pass


class RoadRunnerEngine(CircuitEngine):
    """Engine to simulate gene regulatory networks using RoadRunner.
    
    Args:
        model: Model of the gene regulatory network.
        result: Result of the simulation.
    """
    def __init__(self):
        self.model = None
        self.result = None

    @override
    def copy(self):
        engine = RoadRunnerEngine()
        engine.model = self.model
        return engine

    @override
    def load_circuits(self, *circuits: Circuit):
        """Load engine with model network."""
        encodings = []
        for circuit in circuits:
            encodings.append(self._encode(circuit))
        self.model = "\n".join(encodings)

    @staticmethod
    def _encode(circuit: Circuit) -> str:
        """Encodes a single circuit into a string representation."""
        match circuit:
            case DegFirstOrder(x, k):
                return f"{x} -> ; {k} * {x}"
            case ProdAnd(z, x, y, k, nx, ny, s, a0):
                s = "" if s is None else s
                return f"{s} -> {z}; {a0} + {k} * {x}^{nx} * {y}^{ny} / (1 + {x}^{nx}) / (1 + {y}^{ny})"
            case ProdActivation(y, x, kcat, Km, n, s, a0):
                s = "" if s is None else s
                return f"{s} -> {y}; {a0} + {kcat} * {x}^{n} / ({Km} + {x}^{n})"
            case ProdRepression(y, x, kcat, Km, n, s, a0):
                s = "" if s is None else s
                return f"{s} -> {y}; {a0} + {kcat} / ({Km} + {x}^{n})"

    @override
    def load_sbml(self, sbml):
        pass

    @override
    def step(
        self, 
        metabolite_concs: dict[Gene, float], 
        iter: int = 5, 
        dt: float = 1
    ) -> None:
        """Calculates new gene concentrations."""
        if self.model is None:
            return
        model_full = self._get_model_full(metabolite_concs)

        # Replacement of Tellurium with lower-level functions
        antimony.clearPreviousLoads()
        antimony.freeAll()
        code = antimony.loadAntimonyString(model_full)
        mid = antimony.getMainModuleName()
        rr = roadrunner.RoadRunner(antimony.getSBMLString(mid))

        result = rr.simulate(0, dt, iter)
        self.result = result

    def _get_model_full(self, metabolite_concs: dict[Gene, float]) -> str:
        """Encodes the entire model, including metabolites, 
        into a string representation to be fed into the Roadrunner solver."""
        prefix = "model cell"
        suffix = "end"
        gene_levels = "\n".join(
            [
                f"{metabolite} = {level}"
                for metabolite, level in metabolite_concs.items()
            ]
        )
        return "\n".join([prefix, self.model, gene_levels, suffix])

    @override
    def retrieve_concs(self):
        """Get current gene concentrations."""
        if self.result is None:
            return {}
        colnames = map(lambda name: Gene(name[1:-1]), self.result.colnames[1:])
        new_concs = dict(zip(colnames, self.result[-1][1:]))
        return new_concs


class GeneRegulatoryNetwork:
    """Gene regulatory network to simulate gene expression.

    Args:
        concs: Initial concentrations of genes.
        circuit_engine: Engine to simulate gene regulatory networks
    """
    def __init__(
        self,
        concs: dict = {},
        circuit_engine: CircuitEngine = None,
    ):
        self.concs = concs
        self._circuit_engine = circuit_engine if circuit_engine else RoadRunnerEngine()

    def copy(self) -> "GeneRegulatoryNetwork":
        grn = GeneRegulatoryNetwork(
            self.concs.copy(),
            self._circuit_engine.copy(),
        )
        return grn

    def load_circuits(self, *circuits: Circuit):
        """Load engine with model network."""
        self._circuit_engine.load_circuits(*circuits)

    def update_concs(
        self,
        diffusion_system: DiffusionSystem = None,
        center: Vector = None,
        radius: float = None,
        dt=1,
    ):
        """Update network concentrations based on underlying diffusion systems, 
        cell center and radius."""
        if diffusion_system is not None:
            if center is None or radius is None:
                raise ValueError(
                    "If there is a diffusion system, center or radius \
                        must be supplied as arguments."
                )
            self.update_signaling_concs(diffusion_system, center, radius)
        self.update_metabolite_concs(dt=dt)

    def update_metabolite_concs(self, iter=5, dt=1):
        """Update network concentrations based on the model."""
        self._circuit_engine.step(self.concs, iter=iter, dt=dt)
        new_concs = self._circuit_engine.retrieve_concs()
        self.concs.update(new_concs)

    def update_signaling_concs(
        self, 
        diffusion_system: DiffusionSystem, 
        center: Vector, 
        radius: float
    ):
        """Update network concentrations based on the underlying diffusion system."""
        signaling_concs = diffusion_system.get_ball_concentrations(center, radius)
        self.concs.update(signaling_concs)
