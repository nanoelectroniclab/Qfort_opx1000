from typing import Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    alpha: float = 1.0
    """Amplitude scale factor applied to all gates. Use values != 1.0 to introduce intentional gate imperfections."""
    max_circuit_depth_in_power: int = 4
    """Maximum GST circuit depth = 2^max_circuit_depth_in_power. Default is 4 (depth up to 16)."""
    num_runs: int = 10000
    """Number of repetitions per circuit. Default is 10000."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use the joint or independent flux offset. Default is 'joint'."""
    # Batch analysis parameters (used by 120b(old_main) post-processing)
    batch_id_start: int = 0
    """First node run ID to include in batch analysis."""
    batch_id_end: int = 0
    """Last node run ID (inclusive) to include in batch analysis."""
    batch_alpha_start: float = 0.9
    """Starting alpha value for batch analysis. Default is 0.9."""
    batch_alpha_end: float = 1.3
    """Ending alpha value for batch analysis. Default is 1.3."""
    batch_alpha_step: float = 0.01
    """Alpha step size for batch analysis. Default is 0.01."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
