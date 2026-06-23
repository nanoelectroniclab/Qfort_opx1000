import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode


# ---------------------------------------------------------------------------
## Fit model and helper functions


def cosine_func(x, A, f, offset):
    return A * np.cos(2 * np.pi * f * x) + offset


def find_maximum_cosine(A, f, xmin, xmax):
    """Find the x value where A * cos(2 * pi * f * x) reaches its maximum within [xmin, xmax]."""
    def x_for_max(n):
        return (2 * n * np.pi) / (2 * np.pi * f)

    n_min = np.ceil((2 * np.pi * f * xmin) / (2 * np.pi))
    n_max = np.floor((2 * np.pi * f * xmax) / (2 * np.pi))
    n_values = np.arange(n_min, n_max + 1, 1)
    x_values = x_for_max(n_values)
    x_values_in_range = x_values[(x_values >= xmin) & (x_values <= xmax)]

    if len(x_values_in_range) > 0:
        x_max = x_values_in_range[0]
        max_value = A * np.cos(2 * np.pi * f * x_max)
        return x_max, max_value
    return None, None


def detuning(qubit, flux):
    """Convert a flux pulse amplitude into the equivalent frequency detuning, in MHz."""
    return -1e-6 * qubit.freq_vs_flux_01_quad_term * flux**2


# ---------------------------------------------------------------------------
## Data class


@dataclass
class FitParameters:
    """Stores the Zgate fit results for a single qubit."""

    success: bool
    A: float = None
    f: float = None
    offset: float = None
    Z_id: float = None
    Z_90: float = None
    Z_180: float = None
    Z_270: float = None
    amplitude_z0: float = None
    amplitude_z90: float = None
    amplitude_z180: float = None
    amplitude_z270: float = None


# ---------------------------------------------------------------------------
## Dataset processing


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Attach flux and detuning coordinates to the raw dataset."""
    qubits = node.namespace["qubits"]
    fluxes = node.namespace["fluxes"]

    ds = ds.assign_coords(flux=(["qubit", "flux_int"], np.array([fluxes[q.name] for q in qubits])))
    ds = ds.assign_coords(
        detuning=(["qubit", "flux_int"], np.array([detuning(q, fluxes[q.name]) for q in qubits]))
    )
    ds.detuning.attrs["long_name"] = "Detuning"
    ds.detuning.attrs["units"] = "MHz"
    return ds


# ---------------------------------------------------------------------------
## Fitting


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit a cosine to state vs detuning for each qubit and compute the Zgate flux amplitudes."""
    qubits = node.namespace["qubits"]
    fit_results = {}
    fitted_list = []

    for qubit in qubits:
        x = ds.sel(qubit=qubit.name).detuning.values
        y = ds.sel(qubit=qubit.name).state.values

        A_guess = (y.max() - y.min()) / 2
        f_guess = 1.5 / (x.max() - x.min())
        offset_guess = y.mean()

        try:
            popt, _ = curve_fit(cosine_func, x, y, p0=[A_guess, f_guess, offset_guess])
            A_fit, f_fit, offset_fit = popt

            Z_id, _ = find_maximum_cosine(A_fit, f_fit, x.min(), x.max())
            Z_90 = Z_id + 1 / (4 * f_fit)
            Z_180 = Z_id + 1 / (2 * f_fit)
            Z_270 = Z_id + 3 / (4 * f_fit)

            quad_term = qubit.freq_vs_flux_01_quad_term
            fit_results[qubit.name] = FitParameters(
                success=True,
                A=A_fit,
                f=f_fit,
                offset=offset_fit,
                Z_id=Z_id,
                Z_90=Z_90,
                Z_180=Z_180,
                Z_270=Z_270,
                amplitude_z0=np.sqrt(-1e6 * Z_id / quad_term),
                amplitude_z90=np.sqrt(-1e6 * Z_90 / quad_term),
                amplitude_z180=np.sqrt(-1e6 * Z_180 / quad_term),
                amplitude_z270=np.sqrt(-1e6 * Z_270 / quad_term),
            )
            fitted_list.append(xr.DataArray(cosine_func(x, *popt), dims="flux_int"))
        except RuntimeError:
            node.log(f"Curve fit failed for {qubit.name}")
            fit_results[qubit.name] = FitParameters(success=False)
            fitted_list.append(xr.DataArray(np.full_like(x, np.nan), dims="flux_int"))

    fitted_da = xr.concat(fitted_list, dim="qubit").assign_coords(qubit=[q.name for q in qubits])
    ds_fit = ds.assign(fitted=fitted_da)

    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the Zgate fit results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        if res["success"]:
            log_callable(
                f"Qubit {q}: SUCCESS! "
                f"Z90 amplitude={res['amplitude_z90']:.4f}, Z180 amplitude={res['amplitude_z180']:.4f}"
            )
        else:
            log_callable(f"Qubit {q}: FAIL!")
