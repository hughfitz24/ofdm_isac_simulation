"""
    The transciever device in this OFDM radar configuration. 

    This file simulates the transmitter in the configuration, that sends out an OFDM symbol and receives
    a set of reflections from objects in the wireless channel.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from .config import load_simulation_config
from .plotting import plot_range_doppler_map, plot_signal_space_diagram
from .radar_receiver import receive_target_echoes
from .data_receiver import calculate_symbol_error_rate, receive_ofdm_symbols
from .transmitter import create_ofdm_transmission
from .wireless_channel import (
    create_downlink_channel_response,
    create_received_downlink_signals,
)


def main() -> None:
    project_directory = Path.cwd()
    config_path = project_directory / "simulation.toml"
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / "simulation.toml"

    config = load_simulation_config(config_path)
    transmission = create_ofdm_transmission(config.data, config.radar)

    radar_reception = receive_target_echoes(
        transmission,
        config.radar,
        config.data,
        config.channel,
        config.targets,
        config.processing,
    )

    downlink_channel_response = create_downlink_channel_response(
        transmission.subcarrier_frequencies_hz,
        transmission.symbol_times_s,
        config.downlink,
        config.radar.random_seed + 2,
    )
    downlink_received_symbols = create_received_downlink_signals(
        transmission.transmitted_symbols,
        downlink_channel_response,
        config.downlink,
        config.radar.random_seed + 3,
    )
    data_reception = receive_ofdm_symbols(
        downlink_received_symbols,
        downlink_channel_response,
        config.downlink.snr_db,
        float((abs(transmission.transmitted_symbols) ** 2).mean()),
        config.downlink.noise_enabled,
    )
    symbol_error_rate = calculate_symbol_error_rate(
        transmission.transmitted_symbols,
        data_reception.detected_symbols,
    )

    range_doppler_figure = plot_range_doppler_map(radar_reception.range_doppler, config)
    signal_space_figure = plot_signal_space_diagram(
        transmission.transmitted_symbols,
        data_reception.equalized_symbols,
        transmission.subcarrier_frequencies_hz,
    )

    print(f"OFDM bandwidth: {config.data.bandwidth_hz / 1e6:.2f} MHz")
    print(f"Nominal range resolution: {config.data.range_resolution_m:.2f} m")
    print(f"Maximum range supported by cyclic prefix: {config.data.maximum_cyclic_prefix_range_m:.2f} m")
    print(f"Unambiguous velocity: ±{config.maximum_unambiguous_velocity_m_s:.2f} m/s")
    print(f"Downlink symbol error rate: {symbol_error_rate:.6g}")
    if config.output.save_plots:
        config.output.directory.mkdir(parents=True, exist_ok=True)
        range_doppler_figure.savefig(
            config.output.directory / "ofdm_range_doppler.png",
            dpi=160,
        )
        signal_space_figure.savefig(
            config.output.directory / "ofdm_signal_space.png",
            dpi=160,
        )
    if config.output.show_plots:
        plt.show()
    else:
        plt.close("all")
