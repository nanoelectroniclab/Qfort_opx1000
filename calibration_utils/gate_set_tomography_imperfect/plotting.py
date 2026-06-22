from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_gst_results(batch_results: Dict, qubits: List[AnyTransmon]) -> Dict[str, Figure]:
    """
    Plot all GST batch analysis figures (fidelity and robustness vs alpha).

    Parameters
    ----------
    batch_results : dict
        Output of run_batch_analysis(), containing alpha_list, conditions, gates,
        prep_fidelity, meas_fidelity, gate_fidelity, gate_robustness.
    qubits : list of AnyTransmon
        List of qubits (used for QubitGrid layout).

    Returns
    -------
    dict
        {figure_name: Figure} with 9 figures.
    """
    alpha_list = batch_results["alpha_list"]
    conditions = batch_results["conditions"]
    prep_fidelity = batch_results["prep_fidelity"]
    meas_fidelity = batch_results["meas_fidelity"]
    gate_fidelity = batch_results["gate_fidelity"]
    gate_robustness = batch_results["gate_robustness"]

    figures = {}

    # Use a dummy xr.Dataset-like object for QubitGrid — reuse prep_fidelity shape
    import xarray as xr
    ds_dummy = xr.Dataset(coords={"qubit": [q.name for q in qubits]})

    grid_locations = [q.grid_location for q in qubits]

    def _make_fig(title: str, ylabel: str, data: np.ndarray, key: str) -> Figure:
        grid = QubitGrid(ds_dummy, grid_locations)
        for ax, qubit in grid_iter(grid):
            q_idx = [q.name for q in qubits].index(qubit["qubit"])
            ax.plot(alpha_list, data.T if data.ndim == 2 else data[:, :, q_idx].T,
                    label=conditions)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(r"Gate amplitude scale ($\alpha$)")
            ax.set_title(qubit["qubit"])
            ax.legend(fontsize=7)
        grid.fig.suptitle(title)
        grid.fig.tight_layout()
        figures[key] = grid.fig
        return grid.fig

    # Preparation fidelity
    _make_fig("State Preparation Fidelity", "Fidelity", prep_fidelity, "figure_prep")

    # Measurement fidelity |0>
    _make_fig("Measurement |0> Fidelity", "Fidelity", meas_fidelity[:, 0, :], "figure_meas_0")

    # Measurement fidelity |1>
    _make_fig("Measurement |1> Fidelity", "Fidelity", meas_fidelity[:, 1, :], "figure_meas_1")

    # Gate fidelities
    gate_labels = ["I", "x90", "y90"]
    gate_keys = ["figure_gate_I", "figure_gate_x90", "figure_gate_y90"]
    for g_idx, (label, key) in enumerate(zip(gate_labels, gate_keys)):
        _make_fig(f"Gate '{label}' Fidelity", "Fidelity", gate_fidelity[:, g_idx, :], key)

    # Gate robustness
    robust_keys = ["figure_gate_I_robust", "figure_gate_x90_robust", "figure_gate_y90_robust"]
    for g_idx, (label, key) in enumerate(zip(gate_labels, robust_keys)):
        _make_fig(f"Gate '{label}' Robustness", "Robustness", gate_robustness[:, g_idx, :], key)

    return figures
