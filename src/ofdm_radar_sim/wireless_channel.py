# ===========================================
# wireless_channel.py
# Models the propagation of signals from a transmitter across a wireless channel
# Models both:
#   - Signals reflecting off of objects in the channel back to the transmitter
#   - Signals propagating through the wireless channel to a receiver
# ===========================================
import numpy as np

from .config import (
    SPEED_OF_LIGHT_M_S,
    DownlinkChannelConfig,
    Target,
    WirelessChannelConfig,
)

def calculate_target_channel_gain(
    target: Target,
    channel: WirelessChannelConfig,
    wavelength_m: float,
) -> float:
    """
    Calculates the square root of the monostatic radar power ratio for one target.
    This is
    """
    power_gain = (
        channel.tx_antenna_gain_linear
        * channel.rx_antenna_gain_linear
        * wavelength_m**2
        * target.rcs_m2
        / ((4 * np.pi) ** 3 * target.range_m**4)
    )
    return float(np.sqrt(power_gain))


def create_reflections_from_target(
    transmitted_symbols: np.ndarray,
    subcarrier_frequencies_hz: np.ndarray,
    symbol_times_s: np.ndarray,
    target: Target,
    carrier_frequency_hz: float,
) -> np.ndarray:
    """
    Finds the reflections of a transmitted OFDM symbol off one target. 
    """
    round_trip_delay_s = 2 * target.range_m / SPEED_OF_LIGHT_M_S
    doppler_frequency_hz = 2 * target.velocity_m_s * carrier_frequency_hz / SPEED_OF_LIGHT_M_S
    delay_phase = np.exp(-2j * np.pi * subcarrier_frequencies_hz * round_trip_delay_s)
    doppler_phase = np.exp(2j * np.pi * doppler_frequency_hz * symbol_times_s)
    return transmitted_symbols * doppler_phase[:, None] * delay_phase[None, :]


def add_awgn(
    received_symbols: np.ndarray,
    snr_db: float,
    random_seed: int,
) -> np.ndarray:
    """
    Adds complex AWGN at the requested measured signal-to-noise ratio.
    """
    signal_power_w = float(np.mean(np.abs(received_symbols) ** 2))
    if signal_power_w == 0:
        return received_symbols
    rng = np.random.default_rng(random_seed)
    noise_power_w = signal_power_w / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power_w / 2) * (
        rng.normal(size=received_symbols.shape)
        + 1j * rng.normal(size=received_symbols.shape)
    )
    return received_symbols + noise


def create_reflected_signals(
    reflections_by_target: tuple[np.ndarray, ...],
    targets: tuple[Target, ...],
    channel: WirelessChannelConfig,
    wavelength_m: float,
    random_seed: int,
) -> np.ndarray:
    """
    Transforms a set of transmitted signals by reflecting them off a set of Target objects.
    """
    if len(reflections_by_target) != len(targets):
        raise ValueError("Each target must have one reflected signal")
    if not reflections_by_target:
        raise ValueError("At least one reflected signal is required")

    received_symbols = np.zeros_like(reflections_by_target[0], dtype=complex)
    for reflection, target in zip(reflections_by_target, targets):
        received_symbols += calculate_target_channel_gain(
            target,
            channel,
            wavelength_m,
        ) * reflection
    if channel.noise_enabled:
        received_symbols = add_awgn(received_symbols, channel.snr_db, random_seed)
    return received_symbols


def create_downlink_channel_response(
    subcarrier_frequencies_hz: np.ndarray,
    symbol_times_s: np.ndarray,
    channel: DownlinkChannelConfig,
    random_seed: int,
) -> np.ndarray:
    """
    Creates the downlink channel response over OFDM symbols and subcarriers.

    A Rician channel is one with random fading, distributed about some strong deterministic component.
    The k term models how strong this deterministic component is (0 for purely random, infty for purely deterministic)
    """
    rng = np.random.default_rng(random_seed)
    channel_response = np.zeros(
        (symbol_times_s.size, subcarrier_frequencies_hz.size),
        dtype=complex,
    )
    path_powers = 10 ** (np.asarray(channel.average_path_gains_db) / 10)
    rician_k = 10 ** (channel.rician_k_factor_db / 10)

    for path_index, (delay_s, path_power) in enumerate(
        zip(channel.path_delays_s, path_powers)
    ):
        if path_index == 0:
            los_gain = np.sqrt(path_power * rician_k / (rician_k + 1))
            diffuse_power = path_power / (rician_k + 1)
            los_phase = np.exp(2j * np.pi * rng.random())
            diffuse_gain = np.sqrt(diffuse_power / 2) * (
                rng.normal() + 1j * rng.normal()
            )
            path_gain = los_gain * los_phase + diffuse_gain
            doppler_hz = channel.direct_path_doppler_hz
        else:
            path_gain = np.sqrt(path_power / 2) * (rng.normal() + 1j * rng.normal())
            doppler_hz = rng.uniform(
                -channel.maximum_doppler_hz,
                channel.maximum_doppler_hz,
            )

        delay_phase = np.exp(-2j * np.pi * subcarrier_frequencies_hz * delay_s)
        doppler_phase = np.exp(2j * np.pi * doppler_hz * symbol_times_s)
        channel_response += path_gain * doppler_phase[:, None] * delay_phase[None, :]
    return channel_response


def create_received_downlink_signals(
    transmitted_symbols: np.ndarray,
    channel_response: np.ndarray,
    channel: DownlinkChannelConfig,
    random_seed: int,
) -> np.ndarray:
    """
    Creates symbols received by the downlink user from the transmitted symbols and channel response.

    Applies the channel response, then adds AWGN when enabled in the channel settings.
    """
    if transmitted_symbols.shape != channel_response.shape:
        raise ValueError("Transmitted symbols and downlink channel response must have matching shapes")
    received_symbols = transmitted_symbols * channel_response
    if channel.noise_enabled:
        received_symbols = add_awgn(received_symbols, channel.snr_db, random_seed)
    return received_symbols
