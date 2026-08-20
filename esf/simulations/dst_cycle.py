# -*- coding: UTF-8 -*-

""" This module builds C-rate and state of charge profiles of DST cycles
One DST cycle has a pre-defined pattern which discharge a cell of 10% before
charging it at 1 C-rate until the initial state of charge.
This script support depth of discharge which are multiple of 5%.

Note JPM: I think this script was made to simulate the DST cycles shown in the
reference paper. I wonder if it still works...

"""
import pathlib
import sys
from math import floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esf.models.cycle_counting_algorithm import CycleCounter
from esf.settings.parameters import (
    DST_DATA_FOLDER,
    PRMS_FOLDER,
    ESFParams,
    check_temperature_unit,
    get_example_params,
)
from esf.simulations.degradation import (
    drive_cycle_degradation_calculator,
)

THIS_FOLDER = pathlib.Path(__file__).parent

# DST_DATA_FOLDER / PRMS_FOLDER come from esf.settings.parameters, which
# resolves them inside the package. They were duplicated here with a different
# (repo-relative) expression, so the two could disagree.
__all__ = [
    "DST_DATA_FOLDER",
    "PRMS_FOLDER",
    "DSTCycleDeg",
    "dst_cycles_experimental_data_v_lims",
    "dst_cycles_from_experimental_data",
]

# TODO: save data to file and load it from file to make it possible
#  to skip running the simulations for each dst cycle every time

LAST_DST_CYCLE_NUMBER = {
    "65_75": 8393.25,
    "45_75": 6591.46,
    "25_75": 5226.39,
    "25_85": 5251.05,
    "50_100": 5485.85,
    "40_100": 4986.55,
    "25_100": 4383.85,
}


class DSTCycleDeg:

    def __init__(
        self,
        soc_min: float,
        soc_max: float,
        prms: ESFParams,
        temperature: float,
        number_of_points_pr_test: int = 22,
        peak_finder_resolution: float = 0.1,
        default_fixed_last_cycle: int = 500,
        fixed_last_cycle: bool = False,
        calculate_degradation: bool = True,
        verbose: bool | None = False,
        simulation_kwargs: dict | None = None,
        degradation_kwargs: dict | None = None,
        use_simple_degradation: bool = False,
    ):
        check_temperature_unit(temperature, prms.temperature_unit)

        self.soc_min = soc_min
        self.soc_max = soc_max
        self.prms = prms
        self.temperature = temperature
        self.number_of_points_pr_test = number_of_points_pr_test
        self.default_fixed_last_cycle = default_fixed_last_cycle
        self.fixed_last_cycle = fixed_last_cycle
        self.verbose = verbose
        self.peak_finder_resolution = peak_finder_resolution
        self.simulation_kwargs = simulation_kwargs or {}
        self.degradation_kwargs = degradation_kwargs or {}

        self.cycle_v = []
        self.v_degradation = []

        self.time_v = []
        self.soc_v = []
        self.temp_v = []
        self.c_rate_v = []

        self._run_simulation()

        if calculate_degradation:
            if use_simple_degradation:
                self._calculate_degradation_loop()
            else:
                self._calculate_degradation()

    def run_degradation_calculation(
        self, prms: ESFParams | None = None, verbose=False
    ):
        if prms is not None and verbose:
            print("received prms")
            prms.pprint()
        self._calculate_degradation(prms=prms)

    def _run_simulation(self):
        (
            time_v,
            soc_v,
            temp_v,
            c_rate_v,
        ) = self.simulate_dst(
            self.soc_max,
            self.soc_min,
            self.temperature,
            **self.simulation_kwargs,
        )
        self.time_v = time_v
        self.soc_v = soc_v
        self.temp_v = temp_v
        self.c_rate_v = c_rate_v

        self.cycle_v = self.get_dst_cycle_numbers(
            self.soc_max,
            self.soc_min,
            self.number_of_points_pr_test,
            self.default_fixed_last_cycle,
            fixed_last_cycle=self.fixed_last_cycle,
        )

        if self.verbose:
            df = self.to_frame()
            print()
            print(60 * "=")
            print("DST cycle dataframe:")
            print(60 * "-")

            print(f"{len(self.cycle_v)=}")
            print(f"{len(self.soc_v)=}")
            print(f"{len(self.temp_v)=}")
            print(f"{len(self.c_rate_v)=}")
            print(f"{len(self.time_v)=}")

            print(df.head())
            print(df.tail())
            print(f"{df.shape=}")
            print(f"number of c-rates found: {df['c-rate'].value_counts()}")
            print(60 * "-")

    @staticmethod
    def simulate_dst(soc_max, soc_min, temperature, precision=10, **kwargs):

        # checks if the input are correct
        if soc_min >= soc_max:
            sys.exit("soc_min must be strictly inferior to soc_max")
        elif (soc_max - soc_min) / 5 - int((soc_max - soc_min) / 5) != 0:
            sys.exit(
                "The difference between soc_max and soc_min must be a multiple of 5"
            )

        cycle_duration = 360
        # percentage of a DST cycle after which 5% state of charge is discharged  (half of SoC in the DST)
        perc_dst = 0.6683

        # definition of one C-rate pattern
        delta_t = np.array(
            [18, 28, 12, 8, 16, 24, 12, 8, 16, 24, 12, 8, 16, 36, 8, 24, 8, 32, 8, 42]
        )
        c_rate = np.array(
            [0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -8, -5, 2, -2, 4, 0]
        )

        # number of full DST cycles to perform
        num_dst = floor((soc_max - soc_min) / 10)

        # number of half DST cycle to perform
        if (soc_max - soc_min) / 10 - int((soc_max - soc_min) / 10) == 0.5:
            num_half_dst = 1
        else:
            num_half_dst = 0

        # definition of the C-rate vector of 1 DST cycle
        c_rate_1_dst = []
        for i in range(0, len(delta_t)):
            c_rate_1_dst = c_rate_1_dst + ([c_rate[i]] * delta_t[i] * precision)
        # definition of the state of charge vector of 1 DST cycle
        #  Soc [%] = C_rate * t[s] /3600 * 100
        soc_1_dst = [0]

        for i in range(1, len(c_rate_1_dst)):
            soc_1_dst.append(soc_1_dst[i - 1] + c_rate_1_dst[i] / precision / 36)

        # first DST cycle
        soc_v = soc_1_dst[:]
        c_rate_v = c_rate_1_dst[:]

        # potential next DST cycles
        for i in range(0, num_dst - 1):
            for k in range(0, len(soc_1_dst)):
                soc_v.append((i + 1) * soc_1_dst[-1] + soc_1_dst[k])
            for k in range(0, len(c_rate_1_dst)):
                c_rate_v.append(c_rate_1_dst[k])

        # potential half DST cycle
        if num_half_dst == 1:
            if num_dst == 0:
                # if the range is equal to 5%
                # in that case the first DST cycle is cropped
                soc_v = soc_1_dst[: floor(len(soc_1_dst) * perc_dst)]
                c_rate_v = c_rate_1_dst[: floor(len(soc_1_dst) * perc_dst)]
            else:
                for k in range(0, floor(len(soc_1_dst) * perc_dst)):
                    soc_v.append(num_dst * soc_1_dst[-1] + soc_1_dst[k])
                for k in range(0, floor(len(c_rate_1_dst) * perc_dst)):
                    c_rate_v.append(c_rate_1_dst[k])

        # charge at 1C-rate until initial state of charge
        num_point_half_way = len(soc_v)
        for k in range(0, len(soc_v)):
            c_rate_v.append(1)
            soc_v.append(
                (soc_min - soc_max)
                + k * (num_dst + num_half_dst) * precision / num_point_half_way
            )

        # time vector
        time_v = np.linspace(
            0,
            2 * cycle_duration * (num_dst + num_half_dst * perc_dst),
            (len(c_rate_v)),
        )

        # scaling, starting from SoC_max
        soc_v = [x + soc_max for x in soc_v]

        # setup temperature array
        temp_v = np.ones_like(soc_v) * temperature

        # converting to appropriate units
        soc_v = np.array(soc_v) / 100

        return (
            time_v,
            soc_v,
            temp_v,
            c_rate_v,
        )

    @staticmethod
    def get_dst_cycle_numbers(
        soc_max, soc_min, number_of_points_pr_test, last_cyc=500, fixed_last_cycle=False
    ):
        # x-axis: DST cycles
        if not fixed_last_cycle:
            last_data_cyc_num_key = str(soc_min) + "_" + str(soc_max)
            if last_data_cyc_num_key in LAST_DST_CYCLE_NUMBER:
                last_cyc = floor(LAST_DST_CYCLE_NUMBER[last_data_cyc_num_key])
        cycles_v = np.linspace(0, last_cyc, number_of_points_pr_test)
        return cycles_v

    def _calculate_degradation_loop(
        self, prms: ESFParams | None = None, df: pd.DataFrame | None = None
    ):
        if prms is None:
            prms = self.prms

        alpha_sei = 5.57e-2
        beta_sei = 121.0
        k_1_time_calendar = 4.14e-10
        k_soc_calendar = 1.04
        k_soc_cycling = 1.04
        k_temperature_calendar = 6.93e-2
        k_temperature_cycling = 6.93e-2
        k_1_dod = 1.40e5
        k_2_dod = -5.01e-1
        k_3_dod = -1.23e5

        def time_deg_model(t, k_t=4.11e-10):
            """

            :param t: time in second
            :param k_t:
            :return: 0.2 means end of life of the battery
            """
            return k_t * t

        def soc_stress_model(soc, k_soc=1.01e00, s_ref=0.5):
            return np.exp(k_soc * (soc - s_ref))

        def temp_stress_model(T, k_T=6.71e-02, T_ref=25):
            """
            :param temp: temperature
            :param k_temp:
            :return:
            """
            return np.exp(k_T * (T - T_ref))

        def dod_deg_model(dod, k_d1, k_d2, k_d3):
            """
            :param dod: depth of discharge
            :param k_dod:
            :return:
            """
            return 1.0 / (k_d1 * np.power(dod + 0.001, k_d2) + k_d3)

        def nonlinear_general_model(alpha_sei, beta_sei, deg):
            """
            :param alpha_sei: coefficient alpha of the SEI model
            :param beta_sei: coefficient beta of the SEI model
            :param deg_per_cyc: degradation per cycle
            :return: total degradation after N cycles, between 0 and 1, 0 meaning new battery
            """
            return (
                1 - alpha_sei * np.exp(-beta_sei * deg) - (1 - alpha_sei) * np.exp(-deg)
            )

        def final_degradation_model(
            time_v, soc_v, temp_v, crate_v, delta=0.1, title=""
        ):

            # cycles counting
            cycle_count1 = CycleCounter(
                time_v=time_v,
                soc_v=soc_v,
                temp_v=temp_v,
                crate_v=crate_v,
                delta=delta,
            )

            cycle_count1.rainflow_process()

            soc_mean = cycle_count1.mean_soc  # noqa: F841, NPY002
            arr_dod = cycle_count1.arr_dod
            arr_n = cycle_count1.arr_n
            arr_soc_mean = cycle_count1.arr_soc_mean
            arr_temp = cycle_count1.arr_temp
            arr_time = cycle_count1.arr_time
            arr_crate = cycle_count1.arr_crate  # noqa: F841, NPY002

            st = time_deg_model(arr_time, k_1_time_calendar)
            sc = soc_stress_model(arr_soc_mean, k_soc_calendar)
            tc = temp_stress_model(arr_temp, k_temperature_calendar)
            cal_degradation = np.sum(st * sc * tc)

            st = dod_deg_model(arr_dod, k_1_dod, k_2_dod, k_3_dod)
            sc = soc_stress_model(arr_soc_mean, k_soc_cycling)
            tc = temp_stress_model(arr_temp, k_temperature_cycling)
            cyc_degradation = np.sum(arr_n * st * sc * tc)

            return cal_degradation, cyc_degradation

        sub_cycle_numbers = self.cycle_v
        cal_deg, cyc_deg_per_DST = final_degradation_model(
            time_v=self.time_v,
            soc_v=self.soc_v,
            temp_v=self.temp_v,
            crate_v=self.c_rate_v,
            delta=0.1,
        )

        lin_deg_arr = sub_cycle_numbers * cyc_deg_per_DST + cal_deg
        non_lin_deg_arr = nonlinear_general_model(alpha_sei, beta_sei, lin_deg_arr)
        self.v_degradation = non_lin_deg_arr

    def _calculate_degradation(
        self,
        prms: ESFParams | None = None,
        df: pd.DataFrame | None = None,
    ):
        if df is None:
            df = self.to_frame()
        if prms is None:
            prms = self.prms

        sub_cycle_numbers = self.cycle_v

        out = drive_cycle_degradation_calculator(
            df,
            prms,
            peak_finder_resolution=self.peak_finder_resolution,
            repetitions=1,
            cycle_numbers=sub_cycle_numbers,
            no_of_points_per_test=self.number_of_points_pr_test,
            verbose=self.verbose,
            **self.degradation_kwargs,
        )
        self.v_degradation = out["loss"]
        self.cycle_v = out["n_tot"]

    def to_frame(self):
        return pd.DataFrame(
            {
                "time": self.time_v,
                "soc": self.soc_v,
                "temperature": self.temp_v,
                "c-rate": self.c_rate_v,
            }
        )

    def update_temperature(self, temperature: float):
        self.temperature = temperature
        self.temp_v = np.ones_like(self.temp_v) * temperature

    @property
    def cycle_numbers(self):
        return self.cycle_v

    @property
    def last_cycle(self):
        return self.cycle_v[-1]

    @property
    def soh(self):
        return 100 * (1 - np.array(self.v_degradation))

    @property
    def soh_delta(self):
        return (self.soh - self.soh[0]) / self.soh[0] * 100

    @property
    def soh_delta_last(self):
        return self.soh_delta[-1]

    def plot_dst_deg_model(self, ax, color=None, label=None, fmt="x-", **kwargs):
        soh = 100 * (1 - np.array(self.v_degradation))

        ax.plot(self.cycle_v, soh, fmt, color=color, label=label, **kwargs)
        ax.set_xlabel("DST cycle")
        ax.set_ylabel("State of Health [%]")
        ax.set_title("Degradation model")
        ax.grid(color="k", linestyle=":", linewidth=1, alpha=0.3)

        plt.tight_layout()
        plt.legend()

    def plot_dst_profile(self, ax1, ax2, color=None, label=None, fmt="-", **kwargs):
        ax1.plot(self.time_v, self.c_rate_v, fmt, color=color, label=label, **kwargs)
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("C-rate")
        ax1.set_title("C-rate profile")

        ax2.plot(self.time_v, self.soc_v, fmt, color=color, label=label, **kwargs)
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("State of Charge [%]")
        ax2.set_title("State of charge profile")
        ax1.grid(color="k", linestyle=":", linewidth=1, alpha=0.3)
        ax2.grid(color="k", linestyle=":", linewidth=1, alpha=0.3)
        plt.tight_layout()


def create_color_list(n_colors: int, cmap: str = "tab10"):
    return mpl.colormaps[cmap](np.linspace(0, 1, n_colors))


#: Room temperature assumed for the published DST tests, in degrees celsius.
#: Stated in the paper's figure; see docs/background-notes.md.
DST_EXPERIMENTAL_TEMPERATURE_C = 20


def dst_cycles_experimental_data_v_lims() -> tuple[list[int], list[int]]:
    """The seven SoC windows of the published DST tests.

    Returns:
        ``(soc_min, soc_max)``, each a list of seven values in percent, in the
        order the data files are numbered.
    """
    v_soc_min = [25, 40, 25, 50, 25, 45, 65]
    v_soc_max = [100, 100, 85, 100, 75, 75, 75]
    return v_soc_min, v_soc_max


def dst_cycles_from_experimental_data(data_folder_path=None) -> pd.DataFrame:
    """Load the digitised DST degradation curves from the publication.

    Args:
        data_folder_path: directory holding the ``DST_<min>_<max>.csv`` files.
            Defaults to the copy bundled with the package, so the common case
            needs no argument.

    Returns:
        A long frame with columns ``N`` (DST cycle number), ``SoH`` (percent)
        and ``label`` (``"<min>-<max> @ 20°C"``), all seven windows stacked.

    The values are read off the figures of Xu et al. (2018); see
    :func:`dst_cycles_experimental_data_v_lims` for the windows.
    """
    if data_folder_path is None:
        data_folder_path = DST_DATA_FOLDER
    data_folder_path = pathlib.Path(data_folder_path)

    v_soc_min, v_soc_max = dst_cycles_experimental_data_v_lims()

    data = []
    for soc_min, soc_max in zip(v_soc_min, v_soc_max, strict=True):
        df_single = pd.read_csv(data_folder_path / f"DST_{soc_min}_{soc_max}.csv")
        df_single.columns = ["N", "SoH"]
        df_single["label"] = (
            f"{soc_min}-{soc_max} @ {DST_EXPERIMENTAL_TEMPERATURE_C}°C"
        )
        data.append(df_single)

    return pd.concat(data, axis=0, ignore_index=True)


def plot_dst_experimental_data(ax, data_folder_path):
    v_soc_min = [25, 40, 25, 50, 25, 45, 65]
    v_soc_max = [100, 100, 85, 100, 75, 75, 75]
    colors = ["red", "blue", "green", "orange", "purple", "black", "grey", "brown"]
    data_folder_path = pathlib.Path(data_folder_path)

    for i in range(0, 7):
        dest_path = "DST_" + str(v_soc_min[i]) + "_" + str(v_soc_max[i]) + ".csv"
        path = data_folder_path / pathlib.Path(dest_path)
        df_data_dst = pd.read_csv(path)
        cycles = df_data_dst.iloc[:, 0]
        soh = df_data_dst.iloc[:, 1]
        label = str(v_soc_min[i]) + "-" + str(v_soc_max[i]) + " at 20°C"

        ax.plot(cycles, soh, "x-", label=label, color=colors[i])
        ax.set_xlabel("DST cycle")
        ax.set_ylabel("State of Health [%]")
        ax.set_ylim([60, 100])
        ax.set_xlim([0, 9000])
        plt.xticks(np.arange(0, 9000, 1000))

        ax.set_title("Experimental data")
        ax.grid(color="k", linestyle=":", linewidth=1, alpha=0.3)


def sensitivity_analysis_plot(
    p1="sei_alpha",
    p2="sei_beta",
    n1: int = 3,
    n2: int = 3,
    nd: float = 2.0,
    nd2: float | None = None,
    prms: ESFParams | None = None,
    verbose: bool = False,
):
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib.lines import Line2D

    if nd2 is None:
        nd2 = nd

    print(80 * "=")

    print(
        f"Sensitivity analysis ({p1} vs. {p2}) of DST cycles {n1}x{n2}, delta: {nd} and {nd2}"
    )
    print(80 * "-")

    prms = get_example_params() if prms is None else prms

    soc_min, soc_max = 25, 100

    fig_deg, (ax_1, ax_2) = plt.subplots(
        2,
        1,
        figsize=(6, 10),
        # dpi=100,
        sharex=True,
    )
    fig_deg.suptitle("Sensitivity analysis DST cycles", fontweight="bold")

    c = DSTCycleDeg(soc_min, soc_max, prms=prms, temperature=293.15)
    soh_ref = c.soh
    cycle_numbers = c.cycle_numbers

    prm_1 = prms.get(p1)
    prm_2 = prms.get(p2)
    print(f"prm_1: {prm_1}, prm_2: {prm_2}")
    if verbose:
        print(80 * "-")
        print("Model sets (original):")
        prms.pprint_model_set()
        print(80 * "-")

    if n1 == 1:
        v1 = np.array([prm_1])
    else:
        v1 = np.linspace(prm_1 / nd, prm_1 * nd, n1)

    if n2 == 1:
        v2 = np.array([prm_2])
    else:
        v2 = np.linspace(prm_2 / nd2, prm_2 * nd2, n2)

    c_cycle = cycler(color=mpl.colormaps["tab10"].colors[:n1])
    m_cycle = cycler(marker=Line2D.filled_markers[1 : n2 + 1])
    property_cycle = iter(c_cycle * m_cycle)

    counter = 0
    for y1 in v1:
        for y2 in v2:
            prms.set(p1, y1)
            prms.set(p2, y2)
            if verbose and counter > 0:
                print(80 * "-")
                print(f"Model sets (iteration {counter}):")
                prms.pprint_model_set()
                print(80 * "-")
            counter += 1

            _c = DSTCycleDeg(soc_min, soc_max, prms=prms, temperature=293.15)
            soh = _c.soh

            soh_delta = (soh - soh_ref) / soh_ref * 100
            soh_delta_last = soh_delta[-1]
            soh_delta_sum = np.sum(soh_delta)
            label = f"{p1}={prms.get(p1):.2e}, {p2}={prms.get(p2):.2e}"

            print(f" -> {label}  {soh_delta_last:.3e}  {soh_delta_sum:.3e}")

            line_property = next(property_cycle)
            ax_1.plot(cycle_numbers, soh, label=label, **line_property)
            ax_2.plot(cycle_numbers, soh_delta, label=label, **line_property)

    ax_1.legend(frameon=False)
    ax_2.set_xlabel("Cycle number")
    ax_1.set_ylabel(r"$SoH$" + " [%]")
    ax_2.set_ylabel(r"$\delta_{SoH}$" + " [%]")
    fig_deg.tight_layout()
    fig_deg.align_labels()


def sensitivity_analysis_plot_old(
    p1="sei_alpha", p2="sei_beta", n_k: int = 3, n_d: float = 2.0
):
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib.lines import Line2D

    print(f"Sensitivity analysis ({p1} vs. {p2}) of DST cycles")
    prms = get_example_params()

    soc_min, soc_max = 25, 100

    fig_deg, (ax_1, ax_2) = plt.subplots(
        2,
        1,
        figsize=(6, 10),
        # dpi=100,
        sharex=True,
    )
    fig_deg.suptitle("Sensitivity analysis DST cycles", fontweight="bold")

    c = DSTCycleDeg(soc_min, soc_max, prms=prms, temperature=293.15)
    soh_ref = c.soh
    cycle_numbers = c.cycle_numbers

    k_temp_0 = prms.k_temperature_cycling
    k_time_0 = prms.k_1_time_calendar
    k_soc_0 = prms.k_soc_calendar
    k_high_soc_0 = prms.k_high_soc_cycling
    k_dod_1_0 = prms.k_1_dod
    k_dod_2_0 = prms.k_2_dod
    k_dod_3_0 = prms.k_3_dod
    sei_alpha_0 = prms.sei_alpha
    sei_beta_0 = prms.sei_beta

    k_times = np.linspace(k_time_0 / n_d, k_time_0 * n_d, n_k)
    k_temps = np.linspace(k_temp_0 / n_d, k_temp_0 * n_d, n_k)
    k_socs = np.linspace(k_soc_0 / n_d, k_soc_0 * n_d, n_k)
    k_high_socs = np.linspace(k_high_soc_0 / n_d, k_high_soc_0 * n_d, n_k)
    k_dod_1s = np.linspace(k_dod_1_0 / n_d, k_dod_1_0 * n_d, n_k)
    k_dod_2s = np.linspace(k_dod_2_0 / n_d, k_dod_2_0 * n_d, n_k)
    k_dod_3s = np.linspace(k_dod_3_0 / n_d, k_dod_3_0 * n_d, n_k)

    sei_alphas = np.linspace(sei_alpha_0 / n_d, sei_alpha_0 * n_d, n_k)
    sei_betas = np.linspace(sei_beta_0 / n_d, sei_beta_0 * n_d, n_k)

    parameters = {
        "k_1_time_calendar": k_times,
        "k_temperature_cycling": k_temps,
        "k_temperature_calendar": k_temps,
        "k_soc_calendar": k_socs,
        "k_soc_cycling": k_socs,
        "k_high_soc_cycling": k_high_socs,
        "k_high_soc_calendar": k_high_socs,
        "k_1_dod": k_dod_1s,
        "k_2_dod": k_dod_2s,
        "k_3_dod": k_dod_3s,
        "sei_alpha": sei_alphas,
        "sei_beta": sei_betas,
    }

    c_cycle = cycler(color=mpl.colormaps["tab10"].colors[:n_k])
    m_cycle = cycler(marker=Line2D.filled_markers[1 : n_k + 1])
    property_cycle = iter(c_cycle * m_cycle)

    v1 = parameters[p1]
    v2 = parameters[p2]
    for y1 in v1:
        for y2 in v2:
            prms.set(p1, y1)
            prms.set(p2, y2)
            label = f"{p1}={prms.get(p1):.2e}, {p2}={prms.get(p2):.2e}"
            c = DSTCycleDeg(soc_min, soc_max, prms=prms, temperature=293.15)
            soh = c.soh
            soh_delta = (soh - soh_ref) / soh_ref * 100

            print(f" -> {label}: {soh[-1] - soh_ref[-1]:.3e} ({soh_delta[-1]:.3e} %)")

            line_property = next(property_cycle)
            ax_1.plot(cycle_numbers, soh, label=label, **line_property)
            ax_2.plot(cycle_numbers, soh_delta, label=label, **line_property)

    ax_1.legend(frameon=False)
    ax_2.set_xlabel("Cycle number")
    ax_1.set_ylabel(r"$SoH$" + " [%]")
    ax_2.set_ylabel(r"$\delta_{SoH}$" + " [%]")
    fig_deg.tight_layout()
    fig_deg.align_labels()


def simulate_dst_cycles(
    prms: ESFParams,
    soc_windows: list[tuple] | None = None,
    temperature: float = 298.15,
    plotting: bool = True,
    verbose: bool = False,
    cmap: str = "tab10",
    sim_kwargs: dict | None = None,
    plotting_kwargs: dict | None = None,
    number_of_points_pr_test: int = 22,
):
    original_prms = get_example_params()
    sim_kwargs = sim_kwargs or {}
    plotting_kwargs = plotting_kwargs or {}

    if soc_windows is None:
        soc_windows = [
            (25, 100),
            (40, 100),
            (25, 85),
            (50, 100),
            (25, 75),
            (45, 75),
            (65, 75),
        ]

    if plotting:
        colors = create_color_list(len(soc_windows), cmap=cmap)

        fig_dst, ax_dst = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        ax_dst_1 = ax_dst[0]
        ax_dst_2 = ax_dst[1]

        fig_deg, axs_deg = plt.subplots(
            1, 1, figsize=(10, 5), dpi=100, sharex=True, sharey=True
        )
    else:
        ax_dst_1 = ax_dst_2 = axs_deg = fig_dst = fig_deg = colors = None

    for i, soc_window in enumerate(soc_windows):
        _soc_min, _soc_max = soc_window
        if verbose:
            print(
                f"Calculating DST cycle for {_soc_min} - {_soc_max} at {temperature}°C"
            )
        c = DSTCycleDeg(
            _soc_min,
            _soc_max,
            prms=prms,
            temperature=temperature,
            verbose=verbose,
            number_of_points_pr_test=number_of_points_pr_test,
            **sim_kwargs,
        )
        c_original = DSTCycleDeg(
            _soc_min,
            _soc_max,
            prms=original_prms,
            temperature=temperature,
            verbose=verbose,
            number_of_points_pr_test=number_of_points_pr_test,
            **sim_kwargs,
        )
        if plotting:
            color = colors[i]
            label = f"{_soc_max}-{_soc_min} @ {temperature}°C"
            c.plot_dst_deg_model(
                axs_deg,
                color,
                label=f"{label}-new",
                **plotting_kwargs,
            )
            c_original.plot_dst_deg_model(
                axs_deg,
                color,
                fmt="o-",
                label=f"{label}-original",
                **plotting_kwargs,
            )
            c.plot_dst_profile(
                ax_dst_1, ax_dst_2, color=color, label=label, **plotting_kwargs
            )
    if plotting:
        axs_deg.legend()
        ax_dst_2.legend()
        fig_deg.tight_layout()
        fig_dst.tight_layout()


def simulate_and_fit_dst_cycles(
    prms: ESFParams | None = None,
    verbose: bool = False,
    number_of_points_pr_test: int = 22,
):
    # Taken from model_definition_GPFitting.py:
    if prms is None:
        prms = get_example_params()

    data_folder_path = DST_DATA_FOLDER

    colors = ["red", "blue", "green", "orange", "purple", "black", "grey"]

    v_soc_min = [25, 40, 25, 50, 25, 45, 65]
    v_soc_max = [100, 100, 85, 100, 75, 75, 75]

    fig_dst, ax_dst = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    ax_dst_1 = ax_dst[0]
    ax_dst_2 = ax_dst[1]

    fig_deg, axs_deg = plt.subplots(
        1, 2, figsize=(10, 5), dpi=100, sharex=True, sharey=True
    )
    ax_deg_1 = axs_deg[0]
    ax_deg_2 = axs_deg[1]
    plot_dst_experimental_data(ax_deg_1, data_folder_path=data_folder_path)

    for _soc_min, _soc_max, color in zip(v_soc_min, v_soc_max, colors, strict=False):
        print(f"Calculating DST cycle for {_soc_min} - {_soc_max} at 20°C")
        c = DSTCycleDeg(
            _soc_min,
            _soc_max,
            prms=prms,
            temperature=293.15,
            verbose=verbose,
            number_of_points_pr_test=number_of_points_pr_test,
        )
        label = f"{_soc_max}-{_soc_min} @ 20°C"
        c.plot_dst_deg_model(
            ax_deg_2,
            color,
            label=label,
        )
        c.plot_dst_profile(ax_dst_1, ax_dst_2, color=color, label=label)
    ax_deg_2.legend()
    ax_dst_2.legend()
    fig_deg.tight_layout()
    fig_dst.tight_layout()
