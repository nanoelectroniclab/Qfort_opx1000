from typing import List

import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        d = ds.sel(qubit=qname)

        if "state" in d:
            y_var, y_label = "state", "State population"
            scale = 1.0
        else:
            y_var, y_label = "I", "I [mV]"
            scale = 1e3

        abs_amp_mV = (d.abs_amp * 1e3).values  # (amp,)
        data = d[y_var].values * scale          # (N, amp)

        im = ax.pcolormesh(abs_amp_mV, d.N.values, data, shading="nearest", cmap="RdBu_r")
        grid.fig.colorbar(im, ax=ax, label=y_label)

        if "optimal_amp" in fits:
            opt_mV = float(fits.sel(qubit=qname)["optimal_amp"]) * 1e3
            ax.axvline(opt_mV, color="k", linestyle="--", linewidth=1.5,
                       label=f"opt = {opt_mV:.2f} mV")
            ax.legend(fontsize=7)

        ax.set_xlabel("Amplitude [mV]")
        ax.set_ylabel("Num. Z pulses")
        ax.set_title(qname)

    operation = ds.attrs.get("operation", "")
    grid.fig.suptitle(f"Z gate error amplification {operation}")
    grid.fig.tight_layout()
    return grid.fig
