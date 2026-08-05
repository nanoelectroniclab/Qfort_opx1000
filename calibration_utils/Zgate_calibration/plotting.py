from typing import Dict, List

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit(
    ds: xr.Dataset, qubits: List[AnyTransmon], ds_fit: xr.Dataset, fit_results: Dict
) -> Figure:
    """Plot state vs detuning with the fitted cosine curve and Z-rotation markers for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    qubits_by_name = {q.name: q for q in qubits}

    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q = qubits_by_name[q_name]
        ds_q = ds.sel(qubit=q_name)
        fit_q = fit_results[q_name]

        ax.plot(ds_q.detuning, ds_q.state, marker="o", linestyle="", label="Data")
        if fit_q["success"]:
            ax.plot(ds_q.detuning, ds_fit.sel(qubit=q_name).fitted, label="Fitted")
            for z_value in (fit_q["Z_id"], fit_q["Z_90"], fit_q["Z_180"], fit_q["Z_270"]):
                ax.axvline(x=z_value, color="k", linestyle="--")
        # detuning's sign follows freq_vs_flux_01_quad_term's sign, which varies per qubit,
        # so the sweep can extend either above or below zero — anchor 0 as one bound either way.
        ax.set_xlim(min(0, float(ds_q.detuning.min())), max(0, float(ds_q.detuning.max())))

        def detuning_to_flux(det, q=q):
            return 1e3 * np.sqrt(-1e6 * det / q.freq_vs_flux_01_quad_term)

        def flux_to_detuning(flux, q=q):
            return -1e-6 * (flux / 1e3) ** 2 * q.freq_vs_flux_01_quad_term

        ax2 = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
        ax2.set_xlabel("Flux (mV)")

        ax.set_title(q_name)
        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("State")

    grid.fig.suptitle("Zgate calibration: Ramsey vs flux amplitude")
    grid.fig.tight_layout()
    return grid.fig
