from typing import Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 1000
    """Number of averages to perform. Default is 1000."""
    ref_frequency_MHz: float = 0.0
    """Reference detuning offset added to the swept frequency range, in MHz. Default is 0.0."""
    num_points: int = 100
    """Number of flux/detuning points to sweep. Default is 100."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use the joint or independent flux offset. Default is 'joint'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
