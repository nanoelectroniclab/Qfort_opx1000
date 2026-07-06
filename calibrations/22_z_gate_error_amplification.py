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
from calibration_utils.z_gate_error_amplification import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        Z GATE ERROR AMPLIFICATION
Applies x90 → N × Z(amplitude_scale=a) → x90 → readout for a sweep of amplitude
pre-factors and number-of-pulses values. Amplitude errors accumulate with N, enabling
a precise calibration of the Z gate amplitude.

Prerequisites:
    - Z pulse length calibrated (cryoscope node).
    - Flux point set (qubit.z.flux_point).

Next steps:
    - qubit.z.operations[operation].amplitude is updated in state.

Logic changes vs old_main (22):
- success check: amplitude < hardware_limit (was buggy: used q.xy limits for z gate) → not-at-sweep-edge
- sweep_base_amp stored in node.results for correct abs_amp on data reload
- boolean_to_int() removed: readout_state() returns int, not bool
"""

node = QualibrationNode[Parameters, Quam](
    name="22_z_gate_error_amplification",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q0"]
    # node.parameters.operation = "-z90"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    operation = node.parameters.operation
    n_avg = node.parameters.num_shots

    if node.parameters.amp_factor_step <= 0:
        raise ValueError(f"amp_factor_step must be > 0, got {node.parameters.amp_factor_step}.")
    if node.parameters.min_amp_factor >= node.parameters.max_amp_factor:
        raise ValueError(
            f"min_amp_factor ({node.parameters.min_amp_factor}) must be "
            f"< max_amp_factor ({node.parameters.max_amp_factor})."
        )

    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    N_max = node.parameters.max_number_rabi_pulses_per_sweep
    if operation == "z180":
        N_pi_vec = np.arange(1, N_max, 2).astype(int)
    elif operation in ["z90", "-z90"]:
        N_pi_vec = np.arange(2, N_max, 4).astype(int)
    else:
        raise ValueError(f"Unrecognized operation '{operation}'. Must be 'z180', 'z90', or '-z90'.")
    if len(N_pi_vec) == 0:
        raise ValueError(
            f"max_number_rabi_pulses_per_sweep={N_max} produces an empty N sweep for operation '{operation}'. "
            f"Minimum required: 2 (z180) or 3 (z90/-z90)."
        )

    # Store base amplitudes before the sweep so analysis stays correct after a data reload
    node.namespace["sweep_base_amp"] = {
        q.name: float(q.z.operations[operation].amplitude) for q in qubits
    }
    node.results["sweep_base_amp"] = node.namespace["sweep_base_amp"]

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "N": xr.DataArray(N_pi_vec, attrs={"long_name": "Number of Z pulses"}),
        "amp": xr.DataArray(amps, attrs={"long_name": "Amplitude pre-factor"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)
        npi = declare(int)
        # Per-qubit count to avoid variable conflicts in multiplexed mode
        count = [declare(int) for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(npi, N_pi_vec)):
                    with for_(*from_array(a, amps)):
                        # Reset
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Pulse sequence: x90 → N×Z(a) → x90
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                            qubit.align()
                            with for_(count[i], 0, count[i] < npi, count[i] + 1):
                                qubit.align()
                                qubit.z.play(operation, amplitude_scale=a)
                                qubit.align()
                            qubit.xy.play("x90")
                        align()
                        # Measurement
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
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    if "sweep_base_amp" in node.results:
        node.namespace["sweep_base_amp"] = node.results["sweep_base_amp"]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    plt.show()
    node.results["figures"] = {"raw_fit": fig_raw_fit}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    operation = node.parameters.operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] != "successful":
                continue
            q.z.operations[operation].amplitude = node.results["fit_results"][q.name]["pi_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
