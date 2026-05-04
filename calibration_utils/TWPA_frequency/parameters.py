from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    TWPA_pump_frequency_span_in_mhz: float = 50.0
    """Span of TWPA pump frequency sweep in MHz. Default is 50 MHz."""
    TWPA_pump_frequency_steps: int = 11
    """Number of steps in TWPA pump frequency sweep. Default is 11."""
    TWPA_address: str = "TCPIP::192.168.50.12::INSTR"
    """TWPA instrument address. Default is 'TCPIP::192.168.50.12::INSTR'."""
class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
