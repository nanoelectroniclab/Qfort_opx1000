import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pygsti
import xarray as xr
from pygsti.modelpacks import smq1Q_XYI as std
from picos import Problem
from picos.expressions.variables import HermitianVariable
from picos.expressions.algebra import trace, partial_transpose
from qm.qua import *
from qutip import Qobj

from qualibrate import QualibrationNode


# ---------------------------------------------------------------------------
## Data class

@dataclass
class GSTResults:
    """Stores GST analysis results for a single qubit under TP, CPTP, and Ideal conditions."""

    success: bool
    results: Dict = field(default_factory=dict)
    """Nested dict: condition -> {rho0, meas_op, gate_op} with fidelity, choi, robustness."""


# ---------------------------------------------------------------------------
# pyGSTi experiment setup


def setup_gst_experiment(max_circuit_depth_in_power: int):
    """Build the pyGSTi StandardGSTDesign and return (exp_design, std_model)."""
    max_circuit_length = [2**i for i in range(max(max_circuit_depth_in_power, 0) + 1)]
    std_model = std.target_model()
    exp_design = pygsti.protocols.StandardGSTDesign(
        std_model,
        std.prep_fiducials(),
        std.meas_fiducials(),
        std.germs(),
        max_circuit_length,
    )
    return exp_design, std_model


# ---------------------------------------------------------------------------
## Circuit string parsing utilities (used when building the QUA program)


def parse_gst_circuit_string(circuit_str: str) -> List[str]:
    """
    Parse a pyGSTi circuit string (e.g. 'Gypi2:0(Gxpi2:0)^2@(0)')
    into a list of pulse labels like ['y90', 'x90', 'x90'].
    """
    clean_str = circuit_str.split()[0]
    clean_str = clean_str.replace("@(0)", "")
    clean_str = clean_str.replace("({})", "(I)")
    clean_str = clean_str.replace("{}", "(I)")
    clean_str = clean_str.replace("([])", "(I)")

    token_map = {"Gxpi2:0": "X", "Gypi2:0": "Y", "I": "I"}
    temp_str = clean_str
    for key, token in token_map.items():
        temp_str = temp_str.replace(key, token)

    while "^" in temp_str:
        def expand_match(match):
            return match.group(1) * int(match.group(2))
        temp_str = re.sub(r"\(([^)]+)\)\^(\d+)", expand_match, temp_str)

    temp_str = temp_str.replace("(", "").replace(")", "")

    final_map = {"X": "x90", "Y": "y90", "I": "I"}
    return [final_map[ch] for ch in temp_str if ch in final_map]


def tokenize_gst_circuits(gst_str: List[List[str]]) -> Tuple[List[List[int]], List[int]]:
    """
    Convert pulse label lists into integer token lists for QUA execution.

    Token map: I=0, x90=1, y90=2, x180=3, y180=4, padding=-1.

    Returns (tokenized_circuits, gate_set_length_list).
    """
    gate_set_token = {"I": 0, "x90": 1, "y90": 2, "x180": 3, "y180": 4}
    gate_set_length_list = [len(g) for g in gst_str]
    max_gate_set_length = max(gate_set_length_list)
    tokenized_circuits = []

    for germ in gst_str:
        tokenized = [gate_set_token[g] for g in germ]
        tokenized += [-1] * (max_gate_set_length - len(germ))
        tokenized_circuits.append(tokenized)

    return tokenized_circuits, gate_set_length_list


# ---------------------------------------------------------------------------
## QUA helper (called inside a QUA program context)


def play_tokenized_gst_circuits(tokenized_germ, depth, qubit, alpha: float = 1.0):
    """
    Play a single tokenized GST circuit inside a QUA program.

    Parameters
    ----------
    tokenized_germ : QUA int array variable
        Token list for the current circuit.
    depth : QUA int
        Number of gates to execute.
    qubit : Transmon
        The qubit to drive.
    alpha : float
        Amplitude scale factor (default 1.0 = perfect gate).
    """
    i = declare(int)
    with for_(i, 0, i < depth, i + 1):
        with switch_(tokenized_germ[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(4)
            with case_(1):
                qubit.xy.play("x90", amplitude_scale=alpha)
            with case_(2):
                qubit.xy.play("y90", amplitude_scale=alpha)
            with case_(3):
                qubit.xy.play("x180", amplitude_scale=alpha)
            with case_(4):
                qubit.xy.play("y180", amplitude_scale=alpha)
            with case_(-1):
                pass


# ---------------------------------------------------------------------------
## Dataset processing


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert the averaged state probabilities into integer counts (count0, count1)."""
    n_runs = node.parameters.num_runs
    count1 = (ds.state.values * n_runs).astype(int)
    count0 = n_runs - count1
    ds["count0"] = (("qubit", "germs"), count0)
    ds["count1"] = (("qubit", "germs"), count1)
    return ds


def transform_dataset_to_gst(ds: xr.Dataset, exp_design) -> pygsti.data.DataSet:
    """Convert an xarray Dataset (with count0/count1) into a pyGSTi DataSet."""
    gst_ds = pygsti.data.DataSet(outcome_labels=["0", "1"])
    for i, crc in enumerate(exp_design.all_circuits_needing_data):
        gst_ds.add_count_dict(
            crc,
            {"0": int(ds.count0.values[0, i]), "1": int(ds.count1.values[0, i])},
        )
    return gst_ds


# ---------------------------------------------------------------------------
## GST analysis


def entanglement_robustness(choi_matrix, solver: str = "mosek", **extra_options) -> float:
    """
    Compute the entanglement robustness of a Choi matrix via SDP (picos + mosek).

    Returns a non-negative float; 0 means the channel is separable / ideal.
    """
    state = choi_matrix
    if isinstance(state, Qobj):
        state = state.full()
    SP = Problem()
    gamma = HermitianVariable("gamma", (4, 4))
    rho = HermitianVariable("rho", (4, 4))
    SP.add_constraint((state + gamma) - rho == 0)
    SP.add_constraint(partial_transpose(rho, 0) >> 0)
    SP.add_constraint(gamma >> 0)
    SP.add_constraint(trace(rho) - 1 >> 0)
    SP.set_objective("min", trace(rho) - 1)
    SP.solve(solver=solver, **extra_options)
    return max(SP.value, 0)


def _empty_gst_result_template() -> Dict:
    gate_entry = lambda: {"choi": None, "fidelity": None, "robustness": None}
    meas_entry = lambda: {"povm": None, "fidelity": None}
    rho_entry = lambda: {"density_mx": None, "fidelity": None}
    return {
        cond: {
            "rho0": rho_entry(),
            "meas_op": {"0": meas_entry(), "1": meas_entry()},
            "gate_op": {"I": gate_entry(), "x90": gate_entry(), "y90": gate_entry()},
        }
        for cond in ["TP", "CPTP", "Ideal"]
    }


def run_gst_analysis(
    ds: xr.Dataset,
    exp_design,
    std_model,
    optimizer_params: Dict = None,
    log_callable=None,
) -> Dict:
    """
    Run pyGSTi StandardGST on a single dataset and return the results dict.

    Returns a nested dict: condition -> {rho0, meas_op, gate_op}.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    if optimizer_params is None:
        optimizer_params = {"maxiter": 1000, "tol": 1e-4}

    gst_ds = transform_dataset_to_gst(ds, exp_design)
    gst_data = pygsti.protocols.ProtocolData(exp_design, gst_ds)
    gst_protocol = pygsti.protocols.StandardGST(optimizer=optimizer_params)
    gst_results = gst_protocol.run(gst_data)

    result = _empty_gst_result_template()
    estimate_keys = ["full TP", "CPTPLND", "Target"]
    native_gate_keys = [(), ("Gxpi2", 0), ("Gypi2", 0)]
    np.set_printoptions(suppress=True, precision=6)

    for i, cond in enumerate(result.keys()):
        est_model = gst_results.estimates[estimate_keys[i]].models["stdgaugeopt"]

        # State preparation
        rho_vec = est_model.preps["rho0"]
        rho_est = pygsti.tools.vec_to_stdmx(rho_vec, basis="pp")
        rho_std = pygsti.tools.vec_to_stdmx(std_model.preps["rho0"], basis="pp")
        state_fidelity = pygsti.tools.fidelity(rho_est, rho_std)
        result[cond]["rho0"]["density_mx"] = rho_est
        result[cond]["rho0"]["fidelity"] = state_fidelity
        log_callable(f"[{cond}] State prep fidelity: {state_fidelity:.6f}")

        # Measurement
        for label, effect_vec in est_model.povms["Mdefault"].items():
            mx_est = pygsti.tools.vec_to_stdmx(effect_vec, basis="pp")
            mx_std = pygsti.tools.vec_to_stdmx(std_model.povms["Mdefault"][str(label)], basis="pp")
            meas_fidelity = pygsti.tools.fidelity(mx_est, mx_std)
            result[cond]["meas_op"][str(label)]["povm"] = mx_est
            result[cond]["meas_op"][str(label)]["fidelity"] = meas_fidelity
            log_callable(f"[{cond}] Meas |{label}> fidelity: {meas_fidelity:.6f}")

        # Gates
        for j, gate in enumerate(result[cond]["gate_op"].keys()):
            matrix_ptm = est_model.operations[native_gate_keys[j]].to_dense()
            choi = pygsti.tools.jamiolkowski.jamiolkowski_iso(
                matrix_ptm, op_mx_basis="pp", choi_mx_basis="std"
            )
            infidelity = pygsti.tools.entanglement_infidelity(
                matrix_ptm, std_model.operations[native_gate_keys[j]].to_dense(), "pp"
            )
            robustness = entanglement_robustness(choi)
            result[cond]["gate_op"][gate]["choi"] = choi
            result[cond]["gate_op"][gate]["fidelity"] = 1 - infidelity
            result[cond]["gate_op"][gate]["robustness"] = robustness
            log_callable(f"[{cond}] Gate {gate} infidelity: {infidelity:.6f}, robustness: {robustness:.6f}")

    return result


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode, exp_design, std_model
) -> Tuple[Dict[str, GSTResults], Dict]:
    """
    Run GST analysis on the raw dataset for each qubit.

    Returns (fit_results, raw_results_dict) where:
    - fit_results: {qubit_name: GSTResults}
    - raw_results_dict: the full nested dict saved to node.results
    """
    qubits = node.namespace["qubits"]
    raw_results = {}
    fit_results = {}

    for qubit in qubits:
        q_name = qubit.name
        ds_q = ds.sel(qubit=q_name)
        try:
            result = run_gst_analysis(
                ds_q.expand_dims("qubit"),
                exp_design,
                std_model,
                log_callable=node.log,
            )
            raw_results[q_name] = result
            fit_results[q_name] = GSTResults(success=True, results=result)
        except Exception as e:
            node.log(f"GST analysis failed for {q_name}: {e}")
            raw_results[q_name] = {}
            fit_results[q_name] = GSTResults(success=False, results={})

    return fit_results, raw_results


# ---------------------------------------------------------------------------
## Batch analysis (120b-old): sweep alpha across multiple saved node runs


def run_batch_analysis(node: QualibrationNode, exp_design, std_model) -> Dict:
    """
    Load multiple saved node runs and run GST analysis on each.

    Reads batch_id_start, batch_id_end, batch_alpha_start/end/step from node.parameters.
    Returns a dict with aggregated arrays for plotting.
    """
    assert node.parameters.batch_alpha_step > 0, (
        f"batch_alpha_step must be positive, got {node.parameters.batch_alpha_step}."
    )

    id_list = list(range(node.parameters.batch_id_start, node.parameters.batch_id_end + 1))
    alpha_list = [
        round(a, 10)
        for a in np.arange(
            node.parameters.batch_alpha_start,
            node.parameters.batch_alpha_end + node.parameters.batch_alpha_step / 2,
            node.parameters.batch_alpha_step,
        )
    ]
    assert len(id_list) == len(alpha_list), (
        f"id_list length ({len(id_list)}) does not match alpha_list length ({len(alpha_list)}). "
        f"Check batch_id_start/end and batch_alpha_start/end/step."
    )

    result_dict_list = []

    for run_id in id_list:
        node.log(f"Loading run ID {run_id:05d}")
        loaded = node.load_from_id(run_id)
        ds_ = loaded.results["ds_raw"]
        result = run_gst_analysis(ds_, exp_design, std_model, log_callable=node.log)
        result_dict_list.append(result)

    conditions = ["TP", "CPTP", "Ideal"]
    gates = ["I", "x90", "y90"]

    prep_fidelity = np.array([[d[c]["rho0"]["fidelity"] for d in result_dict_list] for c in conditions])
    prep_density_mx = np.array([[d[c]["rho0"]["density_mx"] for d in result_dict_list] for c in conditions])
    meas_fidelity = np.array(
        [[[d[c]["meas_op"][str(b)]["fidelity"] for d in result_dict_list] for b in [0, 1]] for c in conditions]
    )
    meas_operation_mx = np.array(
        [[[d[c]["meas_op"][str(b)]["povm"] for d in result_dict_list] for b in [0, 1]] for c in conditions]
    )
    gate_fidelity = np.array(
        [[[d[c]["gate_op"][g]["fidelity"] for d in result_dict_list] for g in gates] for c in conditions]
    )
    gate_robustness = np.array(
        [[[d[c]["gate_op"][g]["robustness"] for d in result_dict_list] for g in gates] for c in conditions]
    )
    gate_choi = np.array(
        [[[d[c]["gate_op"][g]["choi"] for d in result_dict_list] for g in gates] for c in conditions]
    )

    return {
        "alpha_list": alpha_list,
        "conditions": conditions,
        "gates": gates,
        "prep_fidelity": prep_fidelity,
        "prep_density_mx": prep_density_mx,
        "meas_fidelity": meas_fidelity,
        "meas_operation_mx": meas_operation_mx,
        "gate_fidelity": gate_fidelity,
        "gate_robustness": gate_robustness,
        "gate_choi": gate_choi,
    }


def log_gst_results(fit_results: Dict[str, GSTResults], log_callable=None):
    """Log a summary of GST fit results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res.success else "FAIL"
        log_callable(f"Qubit {q}: {status}")
        for cond, data in res.results.items():
            for gate, vals in data["gate_op"].items():
                log_callable(
                    f"  [{cond}] {gate} fidelity={vals['fidelity']:.4f}, robustness={vals['robustness']:.4f}"
                )
