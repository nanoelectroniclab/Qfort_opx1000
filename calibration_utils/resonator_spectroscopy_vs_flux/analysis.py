import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from lmfit import Model, Parameter
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis.models import *
# from qualibration_libs.analysis import fit_oscillation


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    min_offset: float
    idle_offset: float
    dv_phi0: float
    phi0_current: float
    m_pH: float


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
        s_idle_offset = f"\tidle offset: {fit_results[q]['idle_offset'] * 1e3:.0f} mV | "
        s_min_offset = f"min offset: {fit_results[q]['min_offset'] * 1e3:.0f} mV | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_idle_offset + s_min_offset + s_freq + s_shift)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Add the current axis of each qubit to the dataset coordinates for plotting
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

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
    # Find the minimum of each frequency line to follow the resonance vs flux
    peak_freq = ds.IQ_abs.idxmin(dim="detuning")
    # Fit to a cosine using the qiskit function: a * np.cos(2 * np.pi * f * t + phi) + offset
    fit_results_da = fit_oscillation(peak_freq.dropna(dim="flux_bias"), "flux_bias")
    fit_results_ds = xr.merge([fit_results_da.rename("fit_results"), peak_freq.rename("peak_freq")])
    # Extract the relevant fitted parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(fit_results_ds, node)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the fit dataset and fit result dictionary."""

    # Ensure that the phase is between -pi and pi
    flux_idle = -fit.sel(fit_vals="phi")
    flux_idle = np.mod(flux_idle + np.pi, 2 * np.pi) - np.pi
    # converting the phase phi from radians to voltage
    flux_idle = flux_idle / fit.sel(fit_vals="f") / 2 / np.pi
    fit = fit.assign_coords(idle_offset=("qubit", flux_idle.fit_results.data))
    fit.idle_offset.attrs = {"long_name": "idle flux bias", "units": "V"}
    # finding the location of the minimum frequency flux point
    flux_min = flux_idle + ((flux_idle < 0) - 0.5) / fit.sel(fit_vals="f")
    flux_min = flux_min * (np.abs(flux_min) < 0.5) + 0.5 * (flux_min > 0.5) - 0.5 * (flux_min < -0.5)
    fit = fit.assign_coords(flux_min=("qubit", flux_min.fit_results.data))
    fit.flux_min.attrs = {"long_name": "minimum frequency flux bias", "units": "V"}
    # finding the frequency as the sweet spot flux
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    freq_shift = fit.peak_freq.sel(flux_bias=flux_idle.fit_results, method="nearest")
    fit = fit.assign_coords(freq_shift=("qubit", freq_shift.data))
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}
    fit = fit.assign_coords(sweet_spot_frequency=("qubit", freq_shift.data + full_freq))
    fit.sweet_spot_frequency.attrs = {
        "long_name": "sweet spot frequency",
        "units": "Hz",
    }
    # m_pH
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    m_pH = (
        1e12
        * 2.068e-15
        / (1 / fit.sel(fit_vals="f"))
        / node.parameters.input_line_impedance_in_ohm
        * attenuation_factor
    )
    # Assess whether the fit was successful or not
    freq_success = np.abs(freq_shift.data) < node.parameters.frequency_span_in_mhz * 1e6
    nan_success = np.isnan(freq_shift.data) | np.isnan(flux_min.fit_results.data) | np.isnan(flux_idle.fit_results.data)
    success_criteria = freq_success & ~nan_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: FitParameters(
            success=fit.sel(qubit=q).success.values.__bool__(),
            resonator_frequency=float(fit.sweet_spot_frequency.sel(qubit=q).values),
            frequency_shift=float(freq_shift.sel(qubit=q).values),
            min_offset=float(flux_min.sel(qubit=q).fit_results.data),
            idle_offset=float(flux_idle.sel(qubit=q).fit_results.data),
            dv_phi0=1 / fit.sel(fit_vals="f", qubit=q).fit_results.data,
            phi0_current=1
            / fit.sel(fit_vals="f", qubit=q).fit_results.data
            * node.parameters.input_line_impedance_in_ohm
            * attenuation_factor,
            m_pH=m_pH.sel(qubit=q).fit_results.data,
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
                method='ampgo' # Try differnet optimize method for better results
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

