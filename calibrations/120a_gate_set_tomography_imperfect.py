# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.gate_set_tomography_imperfect import (
    Parameters,
    GSTResults,
    setup_gst_experiment,
    parse_gst_circuit_string,
    tokenize_gst_circuits,
    play_tokenized_gst_circuits,
    process_raw_dataset,
    fit_raw_data,
    run_batch_analysis,
    log_gst_results,
    plot_gst_results,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# Old imports (kept for reference):
# from quam_libs.components import QuAM, Transmon
# from quam_libs.macros import qua_declaration, active_reset, readout_state
# from quam_libs.lib.save_utils import fetch_results_as_xarray


# %% {Initialisation}
description = """
        GATE SET TOMOGRAPHY (IMPERFECT)

This node performs Gate Set Tomography (GST) using pyGSTi to fully characterize the
single-qubit gate set {I, X(pi/2), Y(pi/2)}. The 'imperfect' variant adds an amplitude
scale factor (alpha) to all gates, allowing intentional gate imperfections to be
introduced and detected.

The GST circuits are pre-computed by pyGSTi, tokenized, and stored in QUA memory.
Each circuit is repeated num_runs times and measured via state discrimination.
Analysis is performed using pyGSTi StandardGST under TP, CPTP, and Ideal constraints,
extracting Choi matrices, gate fidelities, and entanglement robustness for each gate.

A batch analysis mode (120b-old) is also available: set batch_id_start and
batch_id_end to sweep over previously saved runs with different alpha values.

Prerequisites:
    - Having calibrated the qubit (nodes 04b, 06a, 10a/10b).
    - Having calibrated the readout with state discrimination (nodes 07, 08a).
    - Having specified the desired flux point (qubit.z.flux_point).

State update:
    - None (GST is a characterization node, not a calibration node).
"""

node = QualibrationNode[Parameters, Quam](
    name="120a_gate_set_tomography_imperfect",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set parameters locally for debugging. Ignored when run via GUI or graph."""
    node.parameters.qubits = ["q1"]
    # node.parameters.alpha = 1.0
    # node.parameters.max_circuit_depth_in_power = 4
    # node.parameters.num_runs = 10000
    # node.parameters.reset_type = "active"
    node.parameters.simulate = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build GST circuits and generate the QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    alpha = node.parameters.alpha
    n_runs = node.parameters.num_runs
    flux_point = node.parameters.flux_point_joint_or_independent

    # Build pyGSTi experiment design
    exp_design, std_model = setup_gst_experiment(node.parameters.max_circuit_depth_in_power)
    node.namespace["exp_design"] = exp_design
    node.namespace["std_model"] = std_model

    # Convert pyGSTi circuits to QUA tokens
    all_circuit_strs = [s.str for s in exp_design.all_circuits_needing_data]
    all_circuit_labels = [parse_gst_circuit_string(s) for s in all_circuit_strs]
    tokenized_circuits, circuit_depths = tokenize_gst_circuits(all_circuit_labels)
    max_circuit_depth = max(circuit_depths)
    total_circuits = len(tokenized_circuits)

    node.namespace["total_circuits"] = total_circuits
    node.namespace["max_circuit_depth"] = max_circuit_depth

    # Register sweep axes for XarrayDataFetcher
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "germs": xr.DataArray(
            np.arange(total_circuits),
            attrs={"long_name": "GST circuit index"},
        ),
    }

    qubit = qubits[0]  # GST currently supports single-qubit only

    with program() as node.namespace["qua_program"]:
        # New variable declaration
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        # Old: I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)

        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        single_germ_order = declare(int)
        native_gate_order = declare(int)
        tokenized_germs_list = declare(
            int, value=np.array(tokenized_circuits).flatten().tolist()
        )
        single_germ_list = declare(int, size=max_circuit_depth)

        # Set flux point
        node.machine.initialize_qpu(target=qubit, flux_point=flux_point)
        # Old: machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(
                single_germ_order,
                0,
                single_germ_order < total_circuits * max_circuit_depth,
                single_germ_order + max_circuit_depth,
            ):
                # Copy current circuit tokens into single_germ_list
                with for_(native_gate_order, 0, native_gate_order < max_circuit_depth, native_gate_order + 1):
                    assign(
                        single_germ_list[native_gate_order],
                        tokenized_germs_list[single_germ_order + native_gate_order],
                    )

                # Reset qubit
                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                # Old: active_reset(qubit, "readout", max_attempts=15, wait_time=500)
                qubit.align()
                wait(10)

                # Play the GST circuit
                play_tokenized_gst_circuits(single_germ_list, max_circuit_depth, qubit, alpha)

                align()

                # Readout
                qubit.readout_state(state[0])
                # Old: readout_state(qubit, state[0])
                save(state[0], state_st[0])

        with stream_processing():
            n_st.save("n")
            state_st[0].buffer(total_circuits).average().save("state1")


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
                node.parameters.num_runs,
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
    exp_design, std_model = setup_gst_experiment(node.parameters.max_circuit_depth_in_power)
    node.namespace["exp_design"] = exp_design
    node.namespace["std_model"] = std_model


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Run GST analysis and store results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    fit_results, raw_results = fit_raw_data(
        node.results["ds_raw"],
        node,
        node.namespace["exp_design"],
        node.namespace["std_model"],
    )
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    node.results["gst_results"] = raw_results

    log_gst_results(fit_results, log_callable=node.log)
    node.outcomes = {
        q_name: ("successful" if res.success else "failed")
        for q_name, res in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot GST batch analysis figures (fidelity and robustness vs alpha)."""
    if node.parameters.batch_id_end > node.parameters.batch_id_start:
        batch_results = run_batch_analysis(
            node,
            node.namespace["exp_design"],
            node.namespace["std_model"],
        )
        node.results["batch_results"] = batch_results
        figures = plot_gst_results(batch_results, node.namespace["qubits"])
        plt.show()
        node.results["figures"] = figures
    else:
        node.log("Batch analysis skipped (batch_id_start == batch_id_end). Set batch_id_start and batch_id_end to enable.")


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """GST is a characterization node — no qubit parameters are updated here."""
    # old_main restored qubit thread/core assignments here after active reset,
    # because the old quam_libs implementation required deleting the core fields
    # before the QUA program and manually restoring them afterward.
    # The new qubit.reset() handles thread management internally, so nothing is needed here.
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
