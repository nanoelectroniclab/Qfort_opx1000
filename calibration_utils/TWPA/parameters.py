from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""

    # Readout frequency
    frequency_span_in_mhz: float = 30.0
    frequency_step_in_mhz: float = 0.1

    # Readout frequency
    TWPA_pump_power_center_in_dbm: float = 15
    TWPA_pump_power_span_in_dbm: float = 5
    TWPA_pump_power_steps: int = 11

    TWPA_pump_frequency_center_in_mhz: float = 6800.0
    TWPA_pump_frequency_span_in_mhz: float = 200.0
    TWPA_pump_frequency_steps: int = 21

    TWPA_address: str = "TCPIP::192.168.50.12::INSTR"
    """TWPA instrument address. Default is 'TCPIP::192.168.50.12::INSTR'."""
class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
