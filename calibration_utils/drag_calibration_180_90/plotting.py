from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

# Pulse sequences used in the 10a node (options 0 and 1)
_OPTION_LABELS = {
    0: "option 0 (x180–y90)",
    1: "option 1 (y180–x90)",
}


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plot raw DRAG sweep data for each option and mark the fitted intersection alpha.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset (with ``alpha`` coordinate).
    qubits : list of AnyTransmon
        Qubits to plot.
    fits : xr.Dataset
        Fit dataset containing ``optimal_alpha`` per qubit.

    Returns
    -------
    Figure
        Matplotlib figure with one subplot per qubit.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("DRAG calibration (180/90)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plot option 0 and 1 raw curves vs alpha and mark the intersection alpha (no fit curves).
    """
    qname = qubit["qubit"]
    d = ds.sel(qubit=qname)

    if "state" in d:
        y_var = "state"
        y_scale = 1.0
        ylabel = "State population"
    elif "I" in d:
        y_var = "I"
        y_scale = 1e3
        ylabel = "Trans. amp. I [mV]"
    else:
        raise RuntimeError("The dataset must contain either 'state' or 'I' for plotting.")

    alpha = d.alpha.values
    for opt in sorted(d.options.values):
        opt_int = int(opt)
        label = _OPTION_LABELS.get(opt_int, f"option {opt_int}")
        ax.plot(alpha, (d[y_var].sel(options=opt) * y_scale).values, "o-", ms=3, label=label)

    optimal_alpha = float(fit["optimal_alpha"])
    ax.axvline(optimal_alpha, color="r", linestyle="--", linewidth=1.5, label=f"α = {optimal_alpha:.4f}")

    ax.set_xlabel(r"DRAG coefficient $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(qname)
    ax.legend(loc="best", fontsize=8)
