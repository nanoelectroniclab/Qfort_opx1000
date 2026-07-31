import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from .analysis import NUM_STATES, STATE_LABELS


def plot_raw_data_with_fit(ds_raw: xr.Dataset, qubit_pairs, ds_fit: xr.Dataset) -> Figure:
    """Plot the readout confusion matrix of every qubit pair as a heatmap."""
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 4.2), squeeze=False)

    for i, qp in enumerate(qubit_pairs):
        ax = axes[0, i]
        # Stored as conf[measured, prepared]; transposed here so rows read as the prepared state,
        # which is how confusion matrices are conventionally displayed.
        conf = ds_fit["confusion"].sel(qubit_pair=qp.name).values.T

        ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_STATES))
        ax.set_xticklabels(STATE_LABELS)
        ax.set_yticks(range(NUM_STATES))
        ax.set_yticklabels(STATE_LABELS)
        ax.set_xlabel("measured")
        ax.set_ylabel("prepared")
        ax.set_title(f"{qp.name}  |  F = {np.mean(np.diag(conf)):.3f}")

        for prepared in range(NUM_STATES):
            for measured in range(NUM_STATES):
                value = conf[prepared, measured]
                ax.text(
                    measured,
                    prepared,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="w" if value > 0.5 else "k",
                    fontsize=8,
                )

    fig.suptitle("Two-qubit readout confusion matrix")
    fig.tight_layout()
    return fig
