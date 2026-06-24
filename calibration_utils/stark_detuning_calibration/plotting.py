from typing import Dict, List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot the detuning-vs-pulse-count heatmap with the fitted optimal Stark detuning for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Stark detuning calibration")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: Dict[str, str], fit: xr.Dataset = None):
    """Plot one qubit's state-vs-detuning-vs-pulse-count heatmap, marking the fitted optimal detuning."""
    ds_q = ds.sel(qubit=qubit["qubit"])
    z = ds_q.state if "state" in ds_q else ds_q.I
    z.assign_coords(detuning_MHz=ds_q.detuning * 1e-6).plot(
        ax=ax, x="detuning_MHz", y="nb_of_pulses", add_colorbar=False
    )
    if fit is not None:
        ax.axvline(float(fit.optimal_detuning) * 1e-6, color="r")
    ax.set_ylabel("Number of pulses")
    ax.set_xlabel("Detuning [MHz]")
    ax.set_title(qubit["qubit"])
