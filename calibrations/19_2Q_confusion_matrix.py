# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.two_qubit_confusion_matrix import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        TWO-QUBIT READOUT CONFUSION MATRIX

Prepares a qubit pair in each of the four computational basis states |00>, |01>, |10>, |11>,
reads both qubits out simultaneously and counts the outcomes. The resulting 4x4 matrix
quantifies the simultaneous readout error, including readout crosstalk between the two qubits.

Convention: conf[measured, prepared] = P(measured | prepared), so every column sums to 1 and
p_measured = conf @ p_true. Mitigation is inv(conf) with no transpose (see 21b). Note this is
the transpose of the per-qubit qubit.resonator.confusion_matrix, which stores [prepared][measured]
and does need inv(M.T). State discrimination is always used.

Prerequisites:
    - Calibrated single-qubit gates for both qubits in the pair.
    - Calibrated readout with a discrimination threshold for both qubits (07_iq_blobs).

Next steps:
    - The matrix is written to qubit_pair.confusion for readout error mitigation.

Logic changes vs the previous 19 on main (which came from old_main 34_2Q):
- Plot axis labels were swapped and the cell annotations transposed; both fixed.
- Active reset no longer repeats itself 4x with an extra wait; uses qubit.reset(reset_type).
- Dropped the unused plot_raw / measure_leak parameters and the flux_point parameter.
- Added a success criterion: the correct outcome must be the most likely one per prepared state.
- Rejects unconfigured qubit pairs and num_shots < 1 instead of failing with None / silent NaN.
"""

node = QualibrationNode[Parameters, Quam](
    name="19_2Q_confusion_matrix",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubit_pairs = ["q0-q2"]
    # node.parameters.num_shots = 2000
    # node.parameters.reset_type = "active"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    unconfigured = [qp.name for qp in qubit_pairs if qp.qubit_control is None or qp.qubit_target is None]
    if unconfigured:
        raise ValueError(
            f"Qubit pairs {unconfigured} have no qubit_control/qubit_target in state.json. "
            "Set them, or restrict the run with the qubit_pairs parameter."
        )

    n_shots = node.parameters.num_shots
    # Guard against a silent all-NaN confusion matrix from dividing by zero shots
    if n_shots < 1:
        raise ValueError(f"num_shots must be at least 1, got {n_shots}.")

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "shot": xr.DataArray(np.arange(n_shots), attrs={"long_name": "shot index"}),
        "init_state_control": xr.DataArray([0, 1], attrs={"long_name": "prepared control state"}),
        "init_state_target": xr.DataArray([0, 1], attrs={"long_name": "prepared target state"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        control_initial = declare(int)
        target_initial = declare(int)
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)

                with for_(*from_array(control_initial, [0, 1])):
                    with for_(*from_array(target_initial, [0, 1])):
                        for i, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()

                            # Prepare the basis state selected by the two sweep variables
                            with if_(control_initial == 1):
                                qp.qubit_control.xy.play("x180")
                            with if_(target_initial == 1):
                                qp.qubit_target.xy.play("x180")
                            qp.align()

                            qp.qubit_control.readout_state(state_control[i])
                            qp.qubit_target.readout_state(state_target[i])
                            assign(state[i], state_control[i] * 2 + state_target[i])
                            save(state[i], state_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_st[i].buffer(2).buffer(2).buffer(n_shots).save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if fit_result["success"] else "failed")
        for qp_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the confusion matrix of every qubit pair as a heatmap."""
    fig_confusion = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        node.results["ds_fit"],
    )
    plt.show()
    node.results["figures"] = {"confusion": fig_confusion}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Store the measured confusion matrix on the qubit pair."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            qp.confusion = node.results["fit_results"][qp.name]["confusion_matrix"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
