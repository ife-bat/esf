

def test_creating_custom_drive_cycle_and_sim():
    import pandas as pd

    from esf.simulations.cycles import drive_cycle_001

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    df = drive_cycle_001()

    print(80 * "=")
    print("Drive cycle 001")
    print(df.describe().T)
    print(df.head())
    print(df.tail())
    print(80 * "=")

    # out = degradation_calculator(df, prms)
    #
    # print(80 * "=")
    # print("Degradation simulation")
    # print(out.describe().T)
    # print(out.head())
    # print(out.tail())
    # print(80 * "=")


def test_dst_sim_using_original_params(shared_datadir):
    import matplotlib.pyplot as plt
    import pandas as pd

    from esf.settings.parameters import (
        get_example_params_from_original_repo,
    )
    from esf.simulations.dst_cycle import DSTCycleDeg, plot_dst_experimental_data

    prms = get_example_params_from_original_repo()
    temperature = 293.15  # kelvin (20 degC), matching the internal units convention
    verbose = False

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    data_folder_path = shared_datadir / "DST_cycles"
    outdir = shared_datadir / "out"

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
        label = f"{_soc_max}-{_soc_min} @ {temperature - 273.15:.0f}°C"
        print(f"Simulating DST cycles for {label}")
        c = DSTCycleDeg(
            _soc_min,
            _soc_max,
            prms=prms,
            temperature=temperature,
            verbose=verbose,
        )
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

    if not outdir.is_dir():
        outdir.mkdir()
    print(f"output figures are located here: {outdir}")
    fig_dst.savefig(outdir / "dst_sim.png")
    fig_deg.savefig(outdir / "dst_deg.png")
