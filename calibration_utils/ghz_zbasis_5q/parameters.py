from typing import List, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    qubit_quintets: List[List[str]] = [["q2", "q0", "q1", "q3", "q4"]]
    """Each inner list is [center, A, B, C, D] qubit names forming a star-topology GHZ circuit."""
    num_shots: int = 20000
    """Number of single-shot measurements. Default is 20000."""
    reset_type: Literal["thermal", "active"] = "thermal"
    """Qubit reset method. Default is 'thermal'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
