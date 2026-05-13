from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the raw phase data for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        # Create a first x-axis for full_freq_GHz
        ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        # Create a second x-axis for detuning_MHz
        ax2 = ax1.twiny()
        ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (phase)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    legend_handles_labels = None
    for ax, qubit in grid_iter(grid):
        handles_labels = plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
        if legend_handles_labels is None:
            legend_handles_labels = handles_labels

    if legend_handles_labels:
        handles, labels = legend_handles_labels
        dedup = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        grid.fig.legend(dedup.values(), dedup.keys(), loc="upper right")

    grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    def _fit_curves(fit_ds: xr.Dataset):
        """Return list of (label, fitted_data) for each TWPA_amp (or single fit)."""
        if fit_ds is None:
            return []
        if "TWPA_amp" in fit_ds.dims:
            curves = []
            for twpa_amp in fit_ds.TWPA_amp.values:
                params = fit_ds.sel(TWPA_amp=twpa_amp)
                curves.append(
                    (
                        f"fit {twpa_amp:.2f} dBm",
                        lorentzian_dip(
                            ds.detuning,
                            float(params.amplitude.values),
                            float(params.position.values),
                            float(params.width.values) / 2,
                            float(params.base_line.mean().values),
                        ),
                    )
                )
            return curves
        return [
            (
                "fit",
                lorentzian_dip(
                    ds.detuning,
                    float(fit_ds.amplitude.values),
                    float(fit_ds.position.values),
                    float(fit_ds.width.values) / 2,
                    float(fit_ds.base_line.mean().values),
                ),
            )
        ]

    # Create a first x-axis for full_freq_GHz
    (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].IQ_abs / u.mV).plot.line(
        ax=ax, x="full_freq_GHz", hue="TWPA_amp", add_legend=False
    )
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs / u.mV).plot.line(
        ax=ax2, x="detuning_MHz", hue="TWPA_amp", add_legend=False
    )
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the fitted data
    for label, fitted_data in _fit_curves(fit):
        ax2.plot(ds.detuning / u.MHz, fitted_data / u.mV, "--", label=label)
    # Return handles/labels so the figure-level legend can be constructed once
    return ax2.get_legend_handles_labels()

import matplotlib.pyplot as plt


def plot_snr_heatmaps(snr_da, qubits: List[AnyTransmon], best_twpa_power, best_twpa_freq):
    """
    Plots a 2D Heatmap of SNR vs TWPA_power and TWPA_freq for all qubits
    using the physical QubitGrid layout.
    """

    # 2. 建立與晶片實體位置對應的網格
    grid = QubitGrid(snr_da, [q.grid_location for q in qubits])
    
    # 3. 走訪每一個子圖與對應的 Qubit
    for ax, qubit_dict in grid_iter(grid):
        q_name = qubit_dict["qubit"]
        
        # 取出這顆 Qubit 的 2D SNR 資料 (TWPA_power, TWPA_freq)
        snr_2d = snr_da.sel(qubit=q_name)
        
        # 將熱圖畫在指定的子圖 (ax) 上
        # add_colorbar=True 會在每個小圖旁邊加上顏色條，方便獨立觀察
        snr_2d.plot(
            ax=ax,
            x="TWPA_freq",   # 指定 X 軸
            y="TWPA_power",  # 指定 Y 軸
            cmap="viridis",  # 顏色主題 (亮度越高代表 SNR 越好)
            add_colorbar=True 
        )
        
        # 設定標題
        ax.set_title(f"{q_name} SNR Heatmap", fontsize=10)
        
        # 🎁 加碼功能：把「最佳參數」的位置用紅星 ⭐️ 標記出來！
        best_p = best_twpa_power.get(q_name)
        best_f = best_twpa_freq.get(q_name)
        
        if best_p is not None and best_f is not None:
            # 畫上紅色的星星，標示演算法找到的最高點
            ax.plot(best_f, best_p, marker='*', color='red', markersize=12, label="Best Point")
            ax.legend(loc="upper right", fontsize=8)

    # --- 整體排版設定 ---
    grid.fig.suptitle("TWPA Optimization: SNR Heatmaps", fontsize=16, y=1.02)
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    
    return grid.fig