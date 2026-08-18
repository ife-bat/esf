"""Plotting for the data containers in :mod:`esf.io.data`.

Same pattern as :mod:`esf.models.fit_plotting`: the data classes stay focused
on data handling, the matplotlib rendering lives here, and the classes expose
thin delegators. Depends on the data object only by duck typing.
"""

import logging
import warnings

import matplotlib.pyplot as plt

from esf.settings.parameters import (
    CYCLE_UNIT,
    TEMPERATURE_UNIT,
    TIME_UNIT,
    DataType,
)
from esf.settings.units import ureg

logger = logging.getLogger(__name__)


def plot_sample_data(
    data_object,
    y_value="L",
    auto_convert_units=True,
    time_unit=TIME_UNIT,
    temperature_unit=TEMPERATURE_UNIT,
    **kwargs,
):
    """Plot the data."""
    return_figure = kwargs.pop("return_figure", False)
    show = kwargs.pop("show", False)
    layout = kwargs.pop("layout", "constrained")
    figsize = kwargs.pop("figsize", (6, 12))

    fig, axd = plt.subplots(
        6,
        figsize=figsize,
        layout=layout,
    )

    ax_cycle = axd[0]
    ax_dod = axd[1]
    ax_cycle_time = axd[2]
    ax_dod_time = axd[3]
    ax_calendar_temperature = axd[4]
    ax_calendar_soc = axd[5]

    sharex = kwargs.pop("sharex", True)
    if sharex:
        ax_cycle.sharex(ax_dod)
        ax_cycle_time.sharex(ax_dod_time)
        ax_calendar_temperature.sharex(ax_calendar_soc)

    sharey = kwargs.pop("sharey", False)
    if sharey:
        ax_calendar_temperature.sharey(ax_calendar_soc)
        ax_cycle_time.sharey(ax_cycle)
        ax_dod_time.sharey(ax_dod)

    plot_types = {
        DataType.CALENDAR_VS_TEMPERATURE: (
            "t",
            y_value,
            "T",
            ax_calendar_temperature,
            "o-",
        ),
        DataType.CALENDAR_VS_SOC: (
            "t",
            y_value,
            "SoC",
            ax_calendar_soc,
            "s-",
        ),
        DataType.CYCLE_VS_TEMPERATURE: ("N", y_value, "T", ax_cycle, "^-"),
        DataType.CYCLE_VS_DOD: ("N", "DoD", "T", ax_dod, "v-"),
    }
    y_labels = {
        "L": "Loss fraction",
        "SoH": "State of Health",
        "SoC": "State of Charge",
        "DoD": "Depth of Discharge",
    }
    observed_data_types = []
    for key, value in data_object.metadata.data.items():
        logger.debug(f"Processing data for {key} ({value.comment})")

        d = data_object.data[data_object.data["uid"] == key]
        d_type = value.data_type
        observed_data_types.append(d_type)

        if d_type not in plot_types:
            logger.debug(f"Data for {key} has un-supported data type {d_type}. Skipping.")
            continue
        x, y, z, ax, fmt = plot_types[d_type]
        logger.debug(f"  {x=}, {y=}, {z=}, {ax=}, {fmt=}")

        if y not in d.columns:
            logger.debug(f"Data for {key} does not have {y}. Skipping.")
            continue

        if auto_convert_units and d_type in [
            DataType.CALENDAR_VS_TEMPERATURE,
            DataType.CALENDAR_VS_SOC,
            DataType.CYCLE_VS_DOD,
        ]:
            t_unit = ureg(value.time_unit)
            d.loc[:, "t"] = d["t"].map(
                lambda x, t_unit=t_unit: (x * t_unit).to(time_unit).magnitude,
                na_action="ignore",
            )
            logger.debug("  converting time units...")

        if auto_convert_units and d_type in [
            DataType.CALENDAR_VS_TEMPERATURE,
            DataType.CYCLE_VS_TEMPERATURE,
            DataType.CYCLE_VS_DOD,
        ]:
            d = data_object._convert_temperature_data(d, temperature_unit)
            logger.debug("  converting temperature units...")
        if all(d[z].isnull()):
            warnings.warn(
                f"Invalid dataset: could not locate any {z} information for {key} ({value.comment})"
            )
            continue
        for val, ds in d.groupby(z):
            logger.debug(f"Plotting {key}... {value.data_type.value}:{val:0.2f}")
            x_values = ds[x]
            y_values = ds[y]

            if x_values.empty or y_values.empty:
                warnings.warn(
                    f"No data vs {z} found for {key}... {value.data_type.value}:{val:0.2f} ({value.comment})"
                )
                continue

            if x_values.isnull().all():
                warnings.warn(
                    f"Missing {x} information for {key}... {value.data_type.value}:{val:0.2f} "
                    f"({value.comment})"
                )
            if y_values.isnull().all():
                warnings.warn(
                    f"Missing {x} or {y} information for {key}... {value.data_type.value}:{val:0.2f} "
                    f"({value.comment})"
                )
            if any(x_values.isnull()):
                warnings.warn(
                    f"Found some missing {x} information for {key}... {value.data_type.value}:{val:0.2f} "
                    f"({value.comment})"
                )
            if any(y_values.isnull()):
                warnings.warn(
                    f"Found some missing {y} information for {key}... {value.data_type.value}:{val:0.2f} "
                    f"({value.comment})"
                )
            logger.debug(f"  plotting {key}... {value.data_type.value}:{val:0.2f}")
            ax.plot(
                x_values,
                y_values,
                fmt,
                label=f"{key:.4}...  {value.data_type.value}:{val:0.2f}",
            )
        # plotting also the cycling data with x-axis as time
        if d_type in [DataType.CYCLE_VS_DOD, DataType.CYCLE_VS_TEMPERATURE]:
            x = "t"
            if d_type == DataType.CYCLE_VS_DOD:
                ax = ax_dod_time

            elif d_type == DataType.CYCLE_VS_TEMPERATURE:
                ax = ax_cycle_time

            for val, ds in d.groupby(z):
                time_values = ds[x]
                if all(time_values.isnull()):
                    warnings.warn(
                        f"Could not locate any time information for {key}... {value.data_type.value}:{val:0.2f} "
                        f"({value.comment})"
                    )
                elif any(time_values.isnull()):
                    warnings.warn(
                        f"Found some missing time information for {key}... {value.data_type.value}:{val:0.2f} "
                        f"({value.comment})"
                    )
                else:
                    ax.plot(
                        ds[x],
                        ds[y],
                        fmt,
                        label=f"{key:.4}...  {value.data_type.value}:{val:0.2f}",
                    )

    ax_calendar_temperature.set_title("Calendar life")
    ax_calendar_soc.set_xlabel(f"Time [{time_unit}]")
    ax_calendar_soc.set_ylabel(y_labels[y_value])
    ax_calendar_temperature.set_ylabel(y_labels[y_value])

    ax_cycle.set_title("Cycle life")
    ax_cycle.set_ylabel(y_labels[y_value])

    ax_cycle_time.set_title("Cycle life vs time")
    ax_cycle_time.set_ylabel(y_labels[y_value])

    ax_dod.set_xlabel(f"Cycle number [{CYCLE_UNIT}]")
    ax_dod.set_ylabel(y_labels["DoD"])

    ax_dod_time.set_xlabel(f"Time [{time_unit}]")
    ax_dod_time.set_ylabel(y_labels["DoD"])

    def _format_ticks(_ax, tick_formatter=None):
        # This is just for fun and personal growth... does not work as expected
        if tick_formatter is None:
            from matplotlib import ticker

            tick_formatter = ticker.ScalarFormatter()

        _ax.xaxis.set_major_formatter(tick_formatter)
        _ax.yaxis.set_major_formatter(tick_formatter)
        _ax.tick_params(labelsize="small")
        _ax.set_xticklabels(_ax.get_xticks(), weight="bold")
        _ax.set_yticklabels(_ax.get_yticks(), weight="bold")

    legend_title_font_properties = kwargs.pop("legend_title_font_properties", {})
    format_ticks = kwargs.pop("format_ticks", False)

    turn_off_warning = True
    original_log_level = logging.getLogger().getEffectiveLevel()

    for ax in axd:
        if format_ticks:
            _format_ticks(ax)
        if turn_off_warning:
            # original_log_level = logging.getLogger().getEffectiveLevel()
            logging.disable(logging.CRITICAL)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            ncol=1,
            fancybox=False,
            frameon=False,
            shadow=False,
            fontsize="x-small",
            title="Data sets",
            title_fontproperties=legend_title_font_properties,
        )
        if turn_off_warning:
            logging.disable(original_log_level)

    fig.suptitle(
        f"SampleData ({len(data_object.metadata.data)} data sets loaded)",
        fontweight="bold",
    )

    fig.align_labels()

    if return_figure:
        return fig
    if show:
        plt.show()
