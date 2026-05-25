import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from lmfit import Model, Parameter
import matplotlib.pyplot as plt
from qualibrate import QualibrationNode
# from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import convert_IQ_to_V, add_amplitude_and_phase
from quam_config.instrument_limits import instrument_limits
from qualibration_libs.analysis.models import *


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_amp = f"The calibrated {fit_results[q]['operation']} amplitude: {1e3 * fit_results[q]['opt_amp']:.2f} mV (x{fit_results[q]['opt_amp_prefactor']:.2f})\n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    if node.namespace["Rabi_ef"] is not None:
        full_amp = np.array([ds.amp_prefactor * q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]])
    else:
        full_amp = np.array(
            [ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]]
        )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    if node.name == "12b_power_rabi_ef" and hasattr(ds, "I"):
        ds = add_amplitude_and_phase(ds, "amp_prefactor", subtract_slope_flag=True)
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    if node.parameters.max_number_pulses_per_sweep == 1:
        ds_fit = ds.sel(nb_of_pulses=1)
        # Fit the power Rabi oscillations
        if node.parameters.use_state_discrimination:
            fit_vals = fit_oscillation(ds_fit.state, "amp_prefactor")
        else:
            if node.name == "12b_power_rabi_ef":
                fit_vals = fit_oscillation(ds_fit.IQ_abs, "amp_prefactor")
            else:
                fit_vals = fit_oscillation(ds_fit.I, "amp_prefactor")

        ds_fit = xr.merge([ds, fit_vals.rename("fit")])
    else:
        ds_fit = ds
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        if node.parameters.use_state_discrimination:
            ds_fit["data_mean"] = ds.state.mean(dim="nb_of_pulses")
        else:
            ds_fit["data_mean"] = ds.I.mean(dim="nb_of_pulses")
        if (ds.nb_of_pulses.data[0] % 2 == 0 and node.parameters.operation == "x180") or (
            ds.nb_of_pulses.data[0] % 2 != 0 and node.parameters.operation != "x180"
        ):
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
        else:
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    if node.parameters.max_number_pulses_per_sweep == 1:
        # Process the fit parameters to get the right amplitude
        phase = fit.fit.sel(fit_vals="phi") - np.pi * (fit.fit.sel(fit_vals="phi") > np.pi / 2)
        factor = (np.pi - phase) / (2 * np.pi * fit.fit.sel(fit_vals="f"))
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": "factor to get a pi pulse",
            "units": "Hz",
        }
        if node.namespace["Rabi_ef"] is not None:
            current_amps = xr.DataArray(
                [q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]],
                coords=dict(qubit=fit.qubit.data),
            )
        else:
            current_amps = xr.DataArray(
                [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
                coords=dict(qubit=fit.qubit.data),
            )
        opt_amp = factor * current_amps
        fit = fit.assign({"opt_amp": opt_amp})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}

    else:
        current_amps = xr.DataArray(
            [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{node.parameters.operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)
    amp_success = fit.opt_amp < limits[0].max_x180_wf_amplitude
    success_criteria = ~nan_success & amp_success
    fit = fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.__float__(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.__float__(),
            operation=node.parameters.operation,
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results


# Redifine the fitting function here.
def _fix_initial_value(x, da):
    if len(da.dims) == 1:
        return float(x)
    else:
        return x

def fit_oscillation(da, dim):
    """
    Fits an oscillatory model to data along a specified dimension using FFT-based initial guesses.
    This function estimates the frequency, amplitude, and phase of an oscillatory signal in the input
    data array `da` along the given dimension `dim` using the Fast Fourier Transform (FFT) for initial
    parameter guesses. It then fits the data to an oscillatory model of the form:
        y(t) = a * cos(2π * f * t + phi) + offset
    using non-linear least squares optimization.
    Parameters
    ----------
    da : xarray.DataArray
        The input data array containing the oscillatory signal to be fitted.
    dim : str
        The name of the dimension along which to perform the fit.
    Returns
    -------
    xarray.DataArray
        An array containing the fitted parameters for each slice along the specified dimension.
        The output has a new dimension 'fit_vals' with coordinates: ['a', 'f', 'phi', 'offset'],
        corresponding to amplitude, frequency, phase, and offset of the fitted oscillation.
    Notes
    -----
    - The function uses FFT to estimate initial values for frequency, amplitude, and phase.
    - The fitting is performed using a model function (oscillation) and the lmfit library.
    - If the fit fails, diagnostic plots are shown for debugging.
    """

    def get_freq_and_amp_and_phase(da, dim):
        def compute_FFT(x, y):
            N = len(x)
            T = x[1] - x[0]
            yf = np.fft.fft(y)
            xf = np.fft.fftfreq(N, T)
            mask = xf > 0.1
            xf, fft_magnitude = xf[mask], np.abs(yf)[mask]
            idx = np.argmax(fft_magnitude)
            peak_freqs = xf
            peak_amps = 2 * fft_magnitude / N
            peak_phases = np.angle(yf[mask])
            return peak_freqs[idx], peak_amps[idx], peak_phases[idx]

        # Apply the FFT computation across the specified dimension
        def get_fft_param(dat, idx):
            return np.apply_along_axis(
                lambda x: compute_FFT(da[dim].values, x)[idx], -1, dat
            )

        params = [
            xr.apply_ufunc(get_fft_param, da, i, input_core_dims=[[dim], []])
            for i in range(3)
        ]
        params = [_fix_initial_value(p, da) for p in params]
        return [
            p.rename(n)
            for p, n in zip(params, ["freq guess", "amp guess", "phase guess"])
        ]

    freq_guess, amp_guess, phase_guess = get_freq_and_amp_and_phase(da, dim)
    offset_guess = da.mean(dim=dim)

    def apply_fit(x, y, a, f, phi, offset):
        try:
            model = Model(oscillation, independent_vars=["t"])
            fit = model.fit(
                y,
                t=x,
                a=Parameter("a", value=a, min=0),
                f=Parameter(
                    "f", value=f, min=np.abs(0.5 * f), max=np.abs(f * 3 + 1e-3)
                ),
                phi=Parameter("phi", value=phi),
                offset=offset,
                method='least_squares' # Try differnet optimize method for better results
            )
            return np.array([fit.values[k] for k in ["a", "f", "phi", "offset"]])
        except RuntimeError as e:
            print(f"{a=}, {f=}, {phi=}, {offset=}")
            plt.plot(x, oscillation(x, a, f, phi, offset))
            plt.plot(x, y)
            plt.show()
            raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        freq_guess,
        phase_guess,
        offset_guess,
        input_core_dims=[[dim], [dim], [], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(fit_vals=("fit_vals", ["a", "f", "phi", "offset"]))