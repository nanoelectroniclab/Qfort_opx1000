# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.ghz_zbasis_5q import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        GHZ STATE PREPARATION AND Z-BASIS MEASUREMENT (5 QUBITS)

Prepares a 5-qubit GHZ state via a star-topology CZ circuit and measures the
result in the Z basis. State probabilities are corrected for readout errors using
the single-qubit confusion matrices, and GHZ fidelity F = P(00000) + P(11111)
is reported.

Circuit (star topology, first qubit in list = center):
    center:      y90
    satellite X: y90 → Cz(center, X) → -y90   (repeated for A, B, C, D)
Ideal outcome: (|00000⟩ + |11111⟩) / √2

Prerequisites:
    - Single-qubit gates calibrated for all 5 qubits.
    - CZ (cz_unipolar) gates calibrated for all 4 center-satellite pairs.
      Note: 20a must have registered cz_unipolar_pulse in each qubit_control.z.operations;
      if QUA program build fails with KeyError on 'cz_unipolar_pulse', re-run 20a update_state.
    - Readout confusion matrices populated (19_2Q_confusion_matrix or 07_iq_blobs).


Logic changes vs old_main (41c):
- Active reset now covers all 5 qubits (was missing center and qubit_D).
- align() added between consecutive CZ-satellite blocks.
- Confusion matrices stored in node.results so analysis is correct on data reload.
- CZ gate: gates['Cz'].execute() → macros['cz_unipolar'].apply().
- Readout correction inverts conf.T; old_main used inv(conf), underestimating GHZ fidelity.
"""

node = QualibrationNode[Parameters, Quam](
    name="41c_ghz_zbasis_5q",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_quintets = [["q2", "q0", "q1", "q3", "q4"]]
    # node.parameters.num_shots = 5000
    # node.parameters.reset_type = "active"
    pass


node.machine = Quam.load()


# ---------------------------------------------------------------------------
## Helper


class _QubitQuintet:
    """Bundles one center qubit with its 4 CZ-connected satellites."""

    def __init__(self, machine, center, qubit_A, qubit_B, qubit_C, qubit_D):
        self.qubit_center = center
        self.qubit_A = qubit_A
        self.qubit_B = qubit_B
        self.qubit_C = qubit_C
        self.qubit_D = qubit_D
        self.all_qubits = [center, qubit_A, qubit_B, qubit_C, qubit_D]
        self.name = f"{center.name}-{qubit_A.name}-{qubit_B.name}-{qubit_C.name}-{qubit_D.name}"

        pair_map = {}
        for qp in machine.qubit_pairs.values():
            if qp.qubit_control is None or qp.qubit_target is None:
                continue
            key = frozenset({qp.qubit_control, qp.qubit_target})
            pair_map[key] = qp

        self.qubit_pair_A = pair_map.get(frozenset({center, qubit_A}))
        self.qubit_pair_B = pair_map.get(frozenset({center, qubit_B}))
        self.qubit_pair_C = pair_map.get(frozenset({center, qubit_C}))
        self.qubit_pair_D = pair_map.get(frozenset({center, qubit_D}))

        missing = [
            role
            for role, qp in [("A", self.qubit_pair_A), ("B", self.qubit_pair_B),
                               ("C", self.qubit_pair_C), ("D", self.qubit_pair_D)]
            if qp is None
        ]
        if missing:
            raise ValueError(
                f"Quintet {self.name}: no CZ pair found for satellites {missing}. "
                "Check that the pairs are calibrated in state.json."
            )


def _build_quintets(node: QualibrationNode) -> list:
    machine = node.machine
    quintets = []
    for names in node.parameters.qubit_quintets:
        center, a, b, c, d = [machine.qubits[n] for n in names]
        quintets.append(_QubitQuintet(machine, center, a, b, c, d))
    return quintets


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    machine = node.machine
    n_shots = node.parameters.num_shots
    # Guard against a silent all-NaN result from dividing by zero shots
    if n_shots < 1:
        raise ValueError(f"num_shots must be at least 1, got {n_shots}.")

    qubit_quintets = _build_quintets(node)
    node.namespace["qubit_quintets"] = qubit_quintets
    num_quintets = len(qubit_quintets)

    # Store confusion matrices in results so analysis works after data reload
    # np.array() is needed because QuamList has no .tolist()
    node.results["quintet_names"] = [qq.name for qq in qubit_quintets]
    node.results["confusion_matrices"] = {
        qq.name: [np.array(q.resonator.confusion_matrix).tolist() for q in qq.all_qubits]
        for qq in qubit_quintets
    }

    node.namespace["sweep_axes"] = {
        "quintet": xr.DataArray([qq.name for qq in qubit_quintets]),
        "shot": xr.DataArray(np.arange(n_shots), attrs={"long_name": "shot index"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        state = [declare(int) for _ in range(num_quintets)]
        state_st = [declare_stream() for _ in range(num_quintets)]
        s_center = [declare(int) for _ in range(num_quintets)]
        s_A = [declare(int) for _ in range(num_quintets)]
        s_B = [declare(int) for _ in range(num_quintets)]
        s_C = [declare(int) for _ in range(num_quintets)]
        s_D = [declare(int) for _ in range(num_quintets)]

        for ii, qq in enumerate(qubit_quintets):
            for qubit in qq.all_qubits:
                machine.initialize_qpu(target=qubit, flux_point="joint")
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)

                # Reset all 5 qubits
                for qubit in qq.all_qubits:
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()

                # GHZ circuit: y90 on center, then satellite-by-satellite CZ
                qq.qubit_center.xy.play("y90")
                # satellite A
                qq.qubit_A.xy.play("y90")
                qq.qubit_pair_A.macros["cz_unipolar"].apply()
                qq.qubit_A.xy.play("-y90")
                align()
                # satellite B
                qq.qubit_B.xy.play("y90")
                qq.qubit_pair_B.macros["cz_unipolar"].apply()
                qq.qubit_B.xy.play("-y90")
                align()
                # satellite C
                qq.qubit_C.xy.play("y90")
                qq.qubit_pair_C.macros["cz_unipolar"].apply()
                qq.qubit_C.xy.play("-y90")
                align()
                # satellite D
                qq.qubit_D.xy.play("y90")
                qq.qubit_pair_D.macros["cz_unipolar"].apply()
                qq.qubit_D.xy.play("-y90")
                align()

                # Readout all 5 qubits
                qq.qubit_center.readout_state(s_center[ii])
                qq.qubit_A.readout_state(s_A[ii])
                qq.qubit_B.readout_state(s_B[ii])
                qq.qubit_C.readout_state(s_C[ii])
                qq.qubit_D.readout_state(s_D[ii])
                assign(
                    state[ii],
                    s_center[ii] * 16 + s_A[ii] * 8 + s_B[ii] * 4 + s_C[ii] * 2 + s_D[ii],
                )
                save(state[ii], state_st[ii])

        with stream_processing():
            n_st.save("n")
            for ii in range(num_quintets):
                state_st[ii].buffer(n_shots).save(f"state{ii + 1}")


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
    node.namespace["qubit_quintets"] = _build_quintets(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if res["success"] else "failed")
        for qname, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    quintet_names = node.results["quintet_names"]
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        quintet_names,
        node.results["ds_fit"],
    )
    plt.show()
    node.results["figures"] = {"ghz_zbasis": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    pass  # Characterisation only — no state.json updates


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
