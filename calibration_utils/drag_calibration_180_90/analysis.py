import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

_SLOPE_EPS = 1e-9


@dataclass
class FitParameters:
    """Stores the DRAG calibration fit results for a single qubit (Yale 180/90 intersection method)."""

    alpha: float
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

    Returns:
    --------
    None
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_alpha = f"\tDRAG coefficient alpha: {fit_results[q]['alpha']:.4f}\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_alpha)
    pass


def get_sweep_base_alpha_per_qubit(node: QualibrationNode) -> dict[str, float]:
    """
    Base DRAG alpha used when the sweep was executed (alpha_actual = base * alpha_prefactor).

    Prefer values recorded at program creation (``sweep_base_alpha``), not the on-disk
    QUAM state, which may still be 0 when ``alpha_setpoint`` was applied only in memory.
    """
    stored = node.namespace.get("sweep_base_alpha") or node.results.get("sweep_base_alpha")
    out: dict[str, float] = {}
    for q in node.namespace["qubits"]:
        if stored and q.name in stored:
            out[q.name] = float(stored[q.name])
        elif node.parameters.alpha_setpoint is not None:
            out[q.name] = float(node.parameters.alpha_setpoint)
        else:
            out[q.name] = float(q.xy.operations[node.parameters.operation].alpha)
    return out


def _sweep_base_alpha_array(node: QualibrationNode, qubit_coord) -> np.ndarray:
    per_qubit = get_sweep_base_alpha_per_qubit(node)
    return np.array([per_qubit[str(q)] for q in qubit_coord.values])


def _validate_sweep_base_alpha(node: QualibrationNode) -> None:
    per_qubit = get_sweep_base_alpha_per_qubit(node)
    zero_qubits = [q for q, a in per_qubit.items() if a == 0.0]
    if not zero_qubits:
        return
    raise ValueError(
        f"Sweep base alpha is 0 for {zero_qubits}. Hardware/config alpha must be non-zero for "
        f"prefactor sweeps, or set node.parameters.alpha_setpoint and record sweep_base_alpha "
        f"during create_qua_program (see 10a tracked_updates)."
    )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    _validate_sweep_base_alpha(node)
    base = _sweep_base_alpha_array(node, ds.qubit)
    alpha = np.outer(base, ds.alpha_prefactor.values)
    ds = ds.assign_coords(alpha=(["qubit", "alpha_prefactor"], alpha))
    ds.alpha.attrs = {"long_name": "DRAG coefficient alpha"}
    return ds


def _measurement_variable(ds: xr.Dataset, node: QualibrationNode) -> str:
    if node.parameters.use_state_discrimination:
        return "state"
    return "I"


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Yale 180/90 DRAG calibration: linear fit of measurement vs alpha_prefactor for each option,
    then take the intersection of the two lines as the optimal DRAG prefactor.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data (after process_raw_dataset).
    node : QualibrationNode
        Node instance (parameters, qubits).

    Returns:
    --------
    xr.Dataset
        Dataset with polyfit coefficients, intersection_prefactor, and optimal_alpha.
    dict[str, FitParameters]
        Per-qubit fit results; ``alpha`` is the absolute DRAG coefficient.
    """
    ds_fit = ds
    y_name = _measurement_variable(ds, node)
    y = ds[y_name]

    poly = y.polyfit(dim="alpha_prefactor", deg=1)
    coeffs = poly.polyfit_coefficients
    ds_fit["polyfit_coefficients"] = coeffs

    # Intersection of option 0 (x180-y90) and option 1 (y180-x90) lines in prefactor space
    dc0 = coeffs.sel(options=1, degree=0) - coeffs.sel(options=0, degree=0)
    dc1 = coeffs.sel(options=1, degree=1) - coeffs.sel(options=0, degree=1)
    intersection_prefactor = -dc0 / dc1
    ds_fit["intersection_prefactor"] = intersection_prefactor

    _validate_sweep_base_alpha(node)
    base_alpha = xr.DataArray(
        _sweep_base_alpha_array(node, ds.qubit),
        dims=["qubit"],
        coords={"qubit": ds.qubit},
    )
    ds_fit["base_alpha"] = base_alpha
    ds_fit["optimal_alpha"] = intersection_prefactor * base_alpha

    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _as_scalar_bool(da: xr.DataArray) -> bool:
    """Coerce a 0-d or length-1 DataArray to a Python bool."""
    return bool(np.asarray(da).squeeze().item())


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Assess fit success and build per-qubit FitParameters."""
    coeffs = fit.polyfit_coefficients
    dc1 = coeffs.sel(options=1, degree=1) - coeffs.sel(options=0, degree=1)
    nan_success = ~np.isnan(fit.optimal_alpha)
    slope_success = np.abs(dc1) > _SLOPE_EPS
    in_range = (fit.intersection_prefactor >= node.parameters.min_amp_factor) & (
        fit.intersection_prefactor <= node.parameters.max_amp_factor
    )
    success_criteria = nan_success & slope_success & in_range
    for dim in list(success_criteria.dims):
        if dim != "qubit" and success_criteria.sizes[dim] == 1:
            success_criteria = success_criteria.squeeze(dim=dim)
    fit = fit.assign({"success": success_criteria})
    fit_results = {
        q: FitParameters(
            alpha=float(fit.optimal_alpha.sel(qubit=q).item()),
            success=_as_scalar_bool(
                success_criteria.sel(qubit=q) if "qubit" in success_criteria.dims else success_criteria
            ),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}
    return fit, fit_results
