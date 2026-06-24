import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the Stark-detuning fit result for a single qubit."""

    stark_detuning: float
    success: bool


def log_fitted_results(fit_results: Dict[str, FitParameters], log_callable=None):
    """Log the fitted Stark detuning for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res.success else "FAIL"
        log_callable(f"Qubit {q}: {status}, Stark detuning = {res.stark_detuning * 1e-6:.4f} MHz")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ data to volts when not using state discrimination."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
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
    ds_fit = ds
    # Get the average along the number of pulses axis to identify the best pulse amplitude
    if node.parameters.use_state_discrimination:
        ds_fit["averaged_data"] = ds.state.mean(dim="nb_of_pulses")
    else:
        ds_fit["averaged_data"] = ds.I.mean(dim="nb_of_pulses")
    ds_fit["optimal_detuning"] = ds_fit["averaged_data"].idxmin(dim="detuning")

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.optimal_detuning)
    snr_success = (
        np.abs(
            (fit["averaged_data"].min("detuning") - fit["averaged_data"].mean("detuning"))
            / fit["averaged_data"].std("detuning")
        )
        > 2
    )
    success_criteria = ~nan_success & snr_success
    fit = fit.assign({"success": success_criteria})
    fit_results = {
        q: FitParameters(
            stark_detuning=float(fit.sel(qubit=q)["optimal_detuning"]),
            success=bool(fit.sel(qubit=q).success),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}
    return fit, fit_results
