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
from quam.components.pulses import SquarePulse
from quam_config import Quam
from calibration_utils.Zgate_calibration import (
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
# from quam_libs.macros import qua_declaration, active_reset, readout_state

# %% {Description}
description = """
        Z GATE CALIBRATION
Logic changes vs old_main's 21_Zgate_calibration:
    - Flux-sweep and plotting xlim formulas now use abs(quad_term): its sign varies per
      qubit, and the old formula produced NaN for some signs.
    - Fixed parameter typo ref_frequnecy_MHz -> ref_frequency_MHz.
    - Added asserts: num_points >= 2, ref_frequency_MHz >= 0 (avoid silent NaN/inf).

Calibrates the flux pulse amplitude for a physical Z rotation (z90, z180, z270/-z90) on
a single qubit, via a Ramsey variant (x90 - flux pulse - x90 - readout) where the flux
pulse amplitude, not idle time, is swept to accumulate phase. State-vs-detuning is fit
with a cosine to extract the flux amplitudes for each rotation angle.

Prerequisites:
    - Resonator spectroscopy (02a), single-qubit XY gates and Ramsey (04b, 06a), readout
      with state discrimination (07, 08a).
    - Cryoscope (16a/16b), so flux pulses are undistorted.
    - qubit.freq_vs_flux_01_quad_term calibrated and non-zero (09_ramsey_vs_flux_calibration).

State update:
    - qubit.z.operations['z0'/'z90'/'z180'/'-z90'] set to SquarePulse with fitted amplitudes.
"""

node = QualibrationNode[Parameters, Quam](
    name="22_Zgate_calibration", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q0"]
    # node.parameters.num_averages = 1000
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

    n_avg = node.parameters.num_averages
    flux_point = node.parameters.flux_point_joint_or_independent

    # Build the per-qubit flux sweep from the desired detuning range.
    # freq_vs_flux_01_quad_term's sign varies per qubit (depends on the flux bias point),
    # so abs() is used here — only its magnitude matters for the flux amplitude needed.
    assert node.parameters.num_points >= 2, f"num_points must be >= 2, got {node.parameters.num_points}."
    assert node.parameters.ref_frequency_MHz >= 0, (
        "ref_frequency_MHz must be >= 0: the sweep starts at a 0 Hz offset, so a negative value "
        f"would push early sweep points negative, causing sqrt() of a negative number "
        f"(got {node.parameters.ref_frequency_MHz})."
    )
    quad_terms = {qubit.name: qubit.freq_vs_flux_01_quad_term for qubit in qubits}
    frequencies = {
        qubit.name: 1e9 * np.linspace(0, 1.5 / qubit.xy.operations["x180"].length, node.parameters.num_points)
        + node.parameters.ref_frequency_MHz * 1e6
        for qubit in qubits
    }
    fluxes = {qubit.name: np.sqrt(frequencies[qubit.name] / abs(quad_terms[qubit.name])) for qubit in qubits}
    node.namespace["fluxes"] = fluxes

    # Register sweep axes for XarrayDataFetcher
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux_int": xr.DataArray(
            np.arange(node.parameters.num_points), attrs={"long_name": "flux sweep index"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        # New variable declaration
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        # Old: I, I_st, Q, Q_st, _, n_st = qua_declaration(num_qubits=num_qubits)

        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        shots = [declare(int) for _ in range(num_qubits)]
        flux = [declare(fixed) for _ in range(num_qubits)]

        if node.parameters.multiplexed:
            for qubit in qubits:
                node.machine.initialize_qpu(target=qubit, flux_point=flux_point)
                # Old: machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        for i, qubit in enumerate(qubits):
            if not node.parameters.multiplexed:
                node.machine.initialize_qpu(target=qubit, flux_point=flux_point)

            with for_(shots[i], 0, shots[i] < n_avg, shots[i] + 1):
                save(shots[i], n_st)
                with for_each_(flux[i], fluxes[qubit.name].tolist()):
                    # Reset qubit
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    # Old: active_reset(qubit) / wait(thermalization_time)
                    qubit.align()

                    qubit.xy.play("x90")
                    qubit.align()
                    wait(10, qubit.z.name)
                    qubit.z.play(
                        "const",
                        amplitude_scale=flux[i] / qubit.z.operations["const"].amplitude,
                        duration=qubit.xy.operations["x180"].length // 4,
                    )
                    wait(10, qubit.z.name)
                    qubit.align()

                    qubit.xy.play("x90")
                    qubit.align()

                    # Readout
                    qubit.readout_state(state[i])
                    # Old: readout_state(qubit, state[i])
                    save(state[i], state_st[i])
                    qubit.align()

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(node.parameters.num_points).average().save(f"state{i + 1}")


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
                node.parameters.num_averages,
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
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Same flux-sweep reconstruction as create_qua_program — see the Prerequisites note there
    # about freq_vs_flux_01_quad_term needing to be calibrated (non-zero) beforehand.
    assert node.parameters.num_points >= 2, f"num_points must be >= 2, got {node.parameters.num_points}."
    assert node.parameters.ref_frequency_MHz >= 0, (
        f"ref_frequency_MHz must be >= 0, got {node.parameters.ref_frequency_MHz}."
    )
    quad_terms = {qubit.name: qubit.freq_vs_flux_01_quad_term for qubit in qubits}
    frequencies = {
        qubit.name: 1e9 * np.linspace(0, 1.5 / qubit.xy.operations["x180"].length, node.parameters.num_points)
        + node.parameters.ref_frequency_MHz * 1e6
        for qubit in qubits
    }
    node.namespace["fluxes"] = {
        qubit.name: np.sqrt(frequencies[qubit.name] / abs(quad_terms[qubit.name])) for qubit in qubits
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
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
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"], node.results["fit_results"]
    )
    plt.show()
    node.results["figures"] = {
        "raw_fit": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the fitted Zgate flux amplitudes back into the QuAM state."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes[qubit.name] == "failed":
                continue
            fit = node.results["fit_results"][qubit.name]
            length = qubit.xy.operations["x180"].length
            qubit.z.operations["z0"] = SquarePulse(length=length, amplitude=fit["amplitude_z0"])
            qubit.z.operations["z90"] = SquarePulse(length=length, amplitude=fit["amplitude_z90"])
            qubit.z.operations["z180"] = SquarePulse(length=length, amplitude=fit["amplitude_z180"])
            qubit.z.operations["-z90"] = SquarePulse(length=length, amplitude=fit["amplitude_z270"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
