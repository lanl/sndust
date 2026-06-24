from dataclasses import dataclass, field
import os, json


@dataclass
class InputParameters:
    particle_filepath: str
    network_filepath: str


@dataclass
class OutputParameters:
    directory: str
    filename: str = "ivp_sols.npy"
    filepath: str = field(init=False)

    def __post_init__(self):
        self.filepath = os.path.join(self.directory, self.filename)


@dataclass
class SolverParameters:
    abstol: float = 1.0e-6
    reltol: float = 1.0e-6
    select: str = "all"
    fail_action: str = "nothing"


@dataclass
class JitParameters:
    debug: bool = True
    nopython: bool = False
    fastmath: bool = False
    parallel: bool = False
    divzero: str = "python"
    bc: bool = True


@dataclass
class PhysicsParameters:
    gas_gamma: float = 5.0 / 3.0
    terminal_temperature: float = 100.0


@dataclass
class SimulationParameters:
    input: InputParameters = None
    output: OutputParameters = None
    solver: SolverParameters = None
    jit: JitParameters = None
    physics: PhysicsParameters = None

    # def load(self, settings_file: str):
    def __init__(self, settings_file: str):
        with open(settings_file, "r") as jfs:
            settings = json.load(jfs)
            self.input = InputParameters(
                settings["particle_file"], settings["network_file"]
            )
            self.output = OutputParameters(settings["output_directory"])
            self.solver = SolverParameters(
                settings["abs_tol"],
                settings["rel_tol"],
                settings["select"],
                settings["fail_action"],
            )
            self.jit = JitParameters(
                settings["jit_debug"],
                settings["jit_nopython"],
                settings["jit_fastmath"],
                settings["jit_parallel"],
                settings["jit_divzero"],
                settings["jit_bc"],
            )
            self.physics = PhysicsParameters(
                settings["gamma"], settings["terminal_temperature"]
            )
