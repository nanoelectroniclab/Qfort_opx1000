from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages per sweep point. Default is 50."""
    operation: Literal["z180", "z90", "-z90"] = "-z90"
    """Z gate operation to calibrate. Default is '-z90'."""
    min_amp_factor: float = 0.9
    """Minimum amplitude pre-factor (relative to current amplitude). Default is 0.9."""
    max_amp_factor: float = 1.1
    """Maximum amplitude pre-factor. Default is 1.1."""
    amp_factor_step: float = 0.005
    """Step size for amplitude pre-factor sweep. Default is 0.005."""
    max_number_rabi_pulses_per_sweep: int = 50
    """Maximum number of Z gate repetitions per sweep point. Default is 50."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
