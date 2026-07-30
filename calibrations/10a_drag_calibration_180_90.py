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
from calibration_utils.drag_calibration_180_90 import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Initialisation}
description = """
        DRAG PULSE CALIBRATION (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses
successively while varying the DRAG coefficient alpha. After such a sequence, the
qubit is expected to always be in the ground state if the DRAG coefficient has the
correct value. Note that the idea is very similar to what is done in node
04b_power_rabi.

This protocol is described in more details in
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit parameters precisely (nodes 04b_power_rabi.py and 06a_ramsey.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).
    
Next steps before going to the next node:
    - Update the DRAG coefficient (alpha): qubit.xy.operations[operation].alpha.
"""


node = QualibrationNode[Parameters, Quam](
    name="10a_drag_calibration_180_90",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]
    node.parameters.load_data_id = None
    # node.parameters.use_state_discrimination = True
    node.parameters.amp_factor_step = 0.04
    node.parameters.num_shots = 1000
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    node.namespace["tracked_qubits"] = []
    if node.parameters.alpha_setpoint is not None:
        for q in qubits:
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
                q.xy.operations[node.parameters.operation].alpha = node.parameters.alpha_setpoint
                node.namespace["tracked_qubits"].append(q)

    # Alpha actually used in generate_config / QUA (may differ from state.json on disk)
    node.namespace["sweep_base_alpha"] = {
        q.name: float(q.xy.operations[node.parameters.operation].alpha) for q in qubits
    }
    node.results["sweep_base_alpha"] = node.namespace["sweep_base_alpha"]

    n_avg = node.parameters.num_shots  # The number of averages
    options = np.array([0, 1])  # DRAG options: 0 for x180-y90 and 1 for y180-x90
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "options": xr.DataArray(options, attrs={"long_name": "DRAG options"}),
        "alpha_prefactor": xr.DataArray(amps, attrs={"long_name": "DRAG coefficient alpha pre-factor", "units": ""}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        count = declare(int)  # QUA variable for counting the qubit pulses
        opt = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(opt, 0, opt < len(options), opt + 1): 
                    with for_(*from_array(a, amps)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            # Loop for error amplification (perform many qubit pulses)
                            with if_(opt == 0):
                                play("x180" * amp(1, 0, 0, a), qubit.xy.name)
                                play("y90" * amp(a, 0, 0, 1), qubit.xy.name)
                            with else_():
                                play("y180" * amp(a, 0, 0, 1), qubit.xy.name)
                                play("x90" * amp(1, 0, 0, a), qubit.xy.name)
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amps)).buffer(2).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(amps)).buffer(2).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).buffer(2).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)
    # Restore sweep reference alpha from the acquisition run (not fresh state.json)
    if "sweep_base_alpha" in node.results:
        node.namespace["sweep_base_alpha"] = node.results["sweep_base_alpha"]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update QUAM DRAG alpha from the Yale intersection fit.

    After analyse_data, the fitted DRAG coefficient lives in:
      - node.results["fit_results"][qubit_name]["alpha"]  (dict, used here)
      - node.results["ds_fit"].sel(qubit=qubit_name)["optimal_alpha"]  (same value)

    fit_raw_data sets optimal_alpha = intersection_prefactor * base alpha, where
    intersection_prefactor is where the option-0 and option-1 lines cross.
    """

    # Revert the temporary alpha override from create_qua_program (e.g. alpha_setpoint debug)
    for qubit in node.namespace.get("tracked_qubits", []):
        qubit.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            alpha = node.results["fit_results"][q.name]["alpha"]
            q.xy.operations[node.parameters.operation].alpha = alpha


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
