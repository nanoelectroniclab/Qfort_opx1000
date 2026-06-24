# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.stark_detuning_calibration import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# Old imports (kept for reference):
# from quam_libs.components import QuAM
# from quam_libs.macros import qua_declaration, active_reset
# from quam_libs.trackable_object import tracked_updates


# %% {Description}
description = """
        AC STARK-SHIFT CALIBRATION WITH DRAG PULSES (GOOGLE METHOD)
Logic changes vs old_main(09a):
    - No tracked_updates equivalent in the new framework: temporary detuning/alpha override
      is saved via get_raw_value() and restored manually in update_state.
    - Reference-valued attributes (e.g. detuning aliased to another operation) require
      setting to None before writing a plain value, or QuAM raises a ValueError.
    - Added asserts: frequency_step_in_mhz > 0, max_number_pulses_per_sweep >= 1.
    - Added missing flux_point_joint_or_independent parameter (used by initialize_qpu).

Plays an increasing number of x180/-x180 (or x90/-x90) pulse pairs at a swept drive
detuning. The detuning that keeps the qubit in |0> regardless of pulse count is the one
that compensates the AC Stark shift; repeating more pairs amplifies any miscalibration,
sharpening the resonance and pinning down the optimal detuning.
Reference: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Resonator spectroscopy (02a), calibrated x180 (04b, 06a).
    - Readout calibrated with state discrimination (07, 08a) for better SNR (optional).
    - Desired flux bias point set.

State update:
    - qubit.xy.operations[operation].detuning set to the fitted Stark detuning.
    - qubit.xy.operations['x180'].alpha set to DRAG_setpoint, if provided.
"""

node = QualibrationNode[Parameters, Quam](
    name="09a_Stark_Detuning", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q0"]
    # node.parameters.operation = "x90"
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the detuning/pulse-count sweep and generate the QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    operation = node.parameters.operation
    flux_point = node.parameters.flux_point_joint_or_independent

    # Temporarily override each qubit's DRAG alpha and the swept operation's detuning so the
    # measurement starts from a known baseline. Saved here (not via a context manager — the new
    # framework has no equivalent of old_main's tracked_updates) and restored in update_state.
    # get_raw_value() is required (not a plain attribute read) because these fields are often
    # QuAM references (e.g. "#../x180_DragCosine/detuning"); reading the resolved value would
    # permanently break the reference once written back. Writing a plain number over a reference
    # also requires clearing it to None first, or QuAM raises a ValueError.
    original_values = {}
    for qubit in qubits:
        original_values[qubit.name] = {
            "detuning": qubit.xy.operations[operation].get_raw_value("detuning"),
            "alpha": qubit.xy.operations["x180"].get_raw_value("alpha"),
        }
        if node.parameters.DRAG_setpoint is not None:
            qubit.xy.operations["x180"].alpha = None
            qubit.xy.operations["x180"].alpha = node.parameters.DRAG_setpoint
        qubit.xy.operations[operation].detuning = None
        qubit.xy.operations[operation].detuning = 0
    node.namespace["original_values"] = original_values

    # Detuning sweep and pulse-count sweep
    assert node.parameters.frequency_step_in_mhz > 0, (
        f"frequency_step_in_mhz must be positive, got {node.parameters.frequency_step_in_mhz}."
    )
    assert node.parameters.max_number_pulses_per_sweep >= 1, (
        f"max_number_pulses_per_sweep must be >= 1, got {node.parameters.max_number_pulses_per_sweep}."
    )
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)
    n_pi_vec = np.arange(1, node.parameters.max_number_pulses_per_sweep + 1)

    # Register sweep axes for XarrayDataFetcher
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "drive detuning", "units": "Hz"}),
        "nb_of_pulses": xr.DataArray(n_pi_vec, attrs={"long_name": "number of pulse pairs"}),
    }

    with program() as node.namespace["qua_program"]:
        # New variable declaration
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        # Old: I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)
        npi = declare(int)
        count = declare(int)

        for i, qubit in enumerate(qubits):
            node.machine.initialize_qpu(target=qubit, flux_point=flux_point)
            # Old: machine.set_all_fluxes(flux_point=flux_point, target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(npi, n_pi_vec)):
                    with for_(*from_array(df, dfs)):
                        # Reset qubit
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        # Old: active_reset(qubit, "readout") / wait(thermalization_time)

                        update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)
                        with for_(count, 0, count < npi, count + 1):
                            if operation == "x180":
                                qubit.xy.play(operation)
                                qubit.xy.play(operation, amplitude_scale=-1.0)
                            else:
                                qubit.xy.play(operation)
                                qubit.xy.play(operation)
                                qubit.xy.play(operation, amplitude_scale=-1.0)
                                qubit.xy.play(operation, amplitude_scale=-1.0)
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)

                        align()

                        # Readout
                        qubit.readout_state(state[i])
                        # Old: assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                        save(state[i], state_st[i])

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(len(dfs)).buffer(len(n_pi_vec)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw dataset."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data and fitted results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the detuning-vs-pulse-count heatmap with the fitted optimal detuning."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    node.results["figures"] = {
        "raw_fit": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Restore the temporary overrides, then write the fitted Stark detuning back into the QuAM state."""
    operation = node.parameters.operation
    # Restore via the raw (possibly reference-string) value saved in create_qua_program. Setting
    # to None first is required because the current value (set in create_qua_program) is a plain
    # number, and writing a reference string back over a plain number is otherwise rejected too.
    for qubit in node.namespace["qubits"]:
        original = node.namespace["original_values"][qubit.name]
        qubit.xy.operations[operation].detuning = None
        qubit.xy.operations[operation].detuning = original["detuning"]
        qubit.xy.operations["x180"].alpha = None
        qubit.xy.operations["x180"].alpha = original["alpha"]

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes[qubit.name] == "failed":
                continue
            fit = node.results["fit_results"][qubit.name]
            qubit.xy.operations[operation].detuning = None
            qubit.xy.operations[operation].detuning = fit["stark_detuning"]
            if node.parameters.DRAG_setpoint is not None:
                qubit.xy.operations["x180"].alpha = None
                qubit.xy.operations["x180"].alpha = node.parameters.DRAG_setpoint


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
