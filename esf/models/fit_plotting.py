"""Plotting for the fit classes in :mod:`esf.models.fitting`.

The fit classes stay focused on fitting; the (matplotlib) rendering of their
results lives here. Each function takes a fit object and reads its public-ish
attributes, so this module depends on ``fitting`` only by duck typing (no
import), which keeps the dependency one-directional.

The fit classes expose thin ``plot_results`` delegators to these functions.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from esf.settings.parameters import (
    CYCLE_UNIT,
    DOD_UNIT,
    L_UNIT,
    RATE_UNIT,
    SOC_UNIT,
    SOH_UNIT,
    TEMPERATURE_UNIT,
    TIME_UNIT,
)

PLOT_LABELS = {
    "N": f"Cycle number [{CYCLE_UNIT}]",
    "t": f"Time [{TIME_UNIT}]",
    "T": f"Temperature [{TEMPERATURE_UNIT}]",
    "L": f"Loss [{L_UNIT}]",
    "SoH": f"State of Health [{SOH_UNIT}]",
    "SoC": f"State of Charge [{SOC_UNIT}]",
    "DoD": f"Depth of Discharge [{DOD_UNIT}]",
    "rate": f"Rate [{RATE_UNIT}]",
}


# --------------------------------------------------------------------------
# single fit (BaseFit / StressFactorFit / NonlinearFit)
# --------------------------------------------------------------------------
def create_fit_subplots(fit):
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        layout="constrained",
        gridspec_kw={
            "height_ratios": [6, 2],
        },
    )

    title = f"{type(fit).__name__}\n"
    regime_str = "cyc" if fit.is_cycling else "cal"
    if fit._model is not None:
        title += f"{fit._model.name}({regime_str})"
    fig.suptitle(title, fontweight="bold")
    return fig, (ax, ax_err)


def plot_single_fit(fit, *, axes=None, **kwargs):
    fit_object = fit.fit_result.fit_objects

    if not fit_object:
        raise ValueError("No fit results to plot")
    skip_post_process = kwargs.pop("skip_post_process", False)

    if axes is None:
        fig, axes = create_fit_subplots(fit)

    fig, axes, legend_handles = _plot_single_fit_axes(fit, axes=axes, **kwargs)

    if not skip_post_process:
        fig.legend(
            handles=legend_handles,
            loc="outside center right",
            frameon=False,
        )
        fig.align_labels()
    return fig


def _plot_single_fit_axes(
    fit,
    axes=None,
    number=0,
    data=None,
    sim=None,
    sim_long=None,
    initial_plot=True,
    iterate_markers=False,
    iterate_colors=True,
    *args,
    **kwargs,
):

    x_col = fit.get_x_col()
    y_col = fit.get_y_col()

    # Update this so that it can use fit._simulate
    if fit.is_stress_factor_fit:
        x_real = fit._stress_values
        y_real = fit._degradation_rates
        # y_col = fit.get_y_col()
        y_col = fit.sf_col
        y_label = "Stress factor"
        legend_sim = "Simulated"
        legend_real = "Observed"
        x_label = f"{x_col} [${fit.x_unit}$]"
    else:
        if data is None:
            data = fit.data
        x_real = data[x_col]
        y_real = data[y_col]
        z_val = fit.z_val or data[fit.z_col].mean()
        y_label = PLOT_LABELS.get(y_col, y_col)
        z_unit = fit.z_unit
        z_val = fit.z_val or data[fit.z_col].mean()
        legend_sim = rf"${fit.z_col}_{{sim}}={z_val:0.1f}\,{z_unit}$"
        legend_real = rf"${fit.z_col}_{{obs}}={z_val:0.1f}\,{z_unit}$"
        x_label = PLOT_LABELS.get(x_col, x_col)

    if sim is None:
        sim_short = fit._simulate(x=x_real)
    if sim_long is None:
        sim_long = fit._simulate(*args, **kwargs)

    y_sim_points = sim_short[y_col].values
    y_sim_long = sim_long[y_col].values
    x_sim_long = sim_long[x_col].values
    y_real_mean = y_real.mean()
    y_err = (y_real - y_sim_points) / y_real_mean

    if iterate_colors:
        color = fit.color(number)
    else:
        color = fit.color(0)
    if iterate_markers:
        marker = fit.markers[number]
    else:
        marker = fit.markers[0]

    if axes is None:
        post_process_fig = True
        initial_plot = True
        fig, (ax, ax_err) = create_fit_subplots(fit)
    else:
        post_process_fig = False
        ax, ax_err = axes
        fig = ax.get_figure()

    sim_artist = ax.plot(x_sim_long, y_sim_long, label=legend_sim, color=color)
    real_artist = ax.plot(
        x_real,
        y_real,
        marker=marker,
        linestyle="",
        label=legend_real,
        color=color,
        fillstyle="none",
    )

    if initial_plot:
        ax_err.axhline(0, color="gray", linestyle="solid", linewidth=0.8)

    err_artist = ax_err.plot(
        x_real,
        y_err,
        linestyle="--",
        label="$Residuals$",
        color=color,
    )

    if initial_plot:
        ax_err.set_ylabel("Residuals [frac.]")
        ax_err.set_ylim(-0.02, 0.02)
        ax_err.set_yticks([0.0, 0.01])
        ax_err.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    legend_elements = [real_artist[0], sim_artist[0], err_artist[0]]
    if post_process_fig:
        fig.legend(
            loc="outside center right",
            frameon=False,
        )

    return fig, (ax, ax_err), legend_elements


# --------------------------------------------------------------------------
# multi fit (NonlinearMultiFit)
# --------------------------------------------------------------------------
def plot_multi_fit(
    multi,
    title=None,
    figsize=(6, 6),
    legend_loc="center right",
    autoscale_residuals=True,
):
    """Plot the fitting results for all groups of a NonlinearMultiFit.

    Creates a figure with a main plot (all fits) and a residuals plot.
    """
    if not multi.fit_results:
        raise ValueError("No fit results to plot. Call fit() first.")

    fig, (ax_main, ax_residuals) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={
            "height_ratios": [4, 1],  # Main axes is 4 times higher than residuals
        },
    )

    legend_handles = []

    # Plot each group on the same axes with different markers/colors
    max_lengend_char_count = 0
    for i, result in enumerate(multi.fit_results):
        z_val = result["z_val"]
        fit_instance = result["fit_instance"]

        # Get data and simulation for this group
        data = fit_instance.data
        x_col = fit_instance.get_x_col()
        y_col = fit_instance.get_y_col()

        x_real = data[x_col]
        y_real = data[y_col]
        y_real_mean = y_real.mean()

        # Get simulations
        sim = fit_instance._simulate(x=x_real)
        sim_long = fit_instance._simulate()

        # Calculate residuals
        y_sim = sim[y_col].values
        y_sim_long = sim_long[y_col].values
        x_sim_long = sim_long[x_col].values
        y_err = (y_real - y_sim) / y_real_mean

        # Get color and marker for this group
        color = fit_instance.color(i)
        marker = fit_instance.markers[i % len(fit_instance.markers)]

        # Get z value and unit
        z_unit = fit_instance.z_unit

        # Create legends
        legend_sim = rf"${multi.z_col}_{{sim}}={z_val:0.1f}\,{z_unit}$"
        legend_real = rf"${multi.z_col}_{{obs}}={z_val:0.1f}\,{z_unit}$"

        # Update the maximum legend character count (for legend alignment)
        max_lengend_char_count = max(
            max(len(legend_sim), len(legend_real)), max_lengend_char_count
        )

        # Plot on main axes
        ax_main.plot(x_sim_long, y_sim_long, label=legend_sim, color=color)
        ax_main.plot(
            x_real,
            y_real,
            marker=marker,
            linestyle="",
            label=legend_real,
            color=color,
            # fillstyle="none",
        )

        # Plot on residuals axes
        if i == 0:
            ax_residuals.axhline(0, color="gray", linestyle="solid", linewidth=0.8)

        ax_residuals.plot(
            x_real,
            y_err,
            linestyle="--",
            marker=marker,
            markersize=4,
            color=color,
        )
        initial_plot = i == 0
        legend_elements = _multi_legend_builder(
            initial_plot, color, marker, legend_real
        )

        # Add to legend handles
        legend_handles.extend(legend_elements)

    # Set labels and titles
    x_label = fit_instance.get_x_col()
    y_label = fit_instance.get_y_col()

    # Format x_label and y_label using PLOT_LABELS if available
    x_label = PLOT_LABELS.get(x_label, x_label)
    y_label = PLOT_LABELS.get(y_label, y_label)

    ax_residuals.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_residuals.set_ylabel("Residuals [frac.]")

    # Set residuals y-axis limits
    if not autoscale_residuals:
        ax_residuals.set_ylim(-0.02, 0.02)
        ax_residuals.set_yticks([-0.01, 0.0, 0.01])

    # Add a main title
    if title is None:
        title = f"Nonlinear Fit Results ({multi.regime} vs {multi.z_col})"
    if title is not False:
        fig.suptitle(title, fontsize=16)
    legend_width = (max_lengend_char_count + 10) * 0.01

    # Add legend
    fig.legend(
        handles=legend_handles,
        loc=legend_loc,
        frameon=False,
    )

    fig.align_labels()

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=1.0 - legend_width)

    return fig


def _multi_legend_builder(initial_plot, color, marker, legend_real):
    if initial_plot:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                color=color,
                label="obs.",
            ),
            Line2D([0], [0], linestyle="-", color=color, label="sim."),
            Line2D([0], [0], linestyle="--", color=color, label="residuals"),
            Line2D([], [], color="none"),
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color=color,
                label=legend_real,
            ),
        ]
    else:
        legend_elements = [
            Line2D(
                [0],
                [0],
                linestyle="",
                marker=marker,
                color=color,
                label=legend_real,
            ),
        ]

    return legend_elements
