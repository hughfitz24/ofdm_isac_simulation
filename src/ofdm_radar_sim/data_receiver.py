from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OFDMReception:
    equalized_symbols: np.ndarray
    detected_symbols: np.ndarray
    noise_variance: float


def estimate_noise_variance(received_symbols: np.ndarray, snr_db: float) -> float:
    """Estimate complex noise power from the received power and configured SNR."""
    snr_linear = 10 ** (snr_db / 10)
    received_power = float(np.mean(np.abs(received_symbols) ** 2))
    return received_power / (snr_linear + 1)


def mmse_equalize(
    received_symbols: np.ndarray,
    channel_response: np.ndarray,
    noise_variance: float,
    symbol_energy: float,
) -> np.ndarray:
    """Apply a per-resource-element linear MMSE equalizer using known channel gains."""
    if received_symbols.shape != channel_response.shape:
        raise ValueError("Received symbols and channel response must have matching shapes")
    if noise_variance < 0 or not np.isfinite(noise_variance):
        raise ValueError("Noise variance must be finite and nonnegative")
    if symbol_energy <= 0 or not np.isfinite(symbol_energy):
        raise ValueError("Symbol energy must be finite and positive")

    channel_power = np.abs(channel_response) ** 2
    regularization = noise_variance / symbol_energy
    denominator = channel_power + regularization
    equalizer = np.divide(
        np.conj(channel_response),
        denominator,
        out=np.zeros_like(channel_response, dtype=complex),
        where=denominator > 0,
    )
    return equalizer * received_symbols


def detect_qpsk_symbols(
    equalized_symbols: np.ndarray,
    symbol_energy: float,
) -> np.ndarray:
    """Map equalized symbols to the nearest transmitted QPSK constellation point."""
    amplitude = np.sqrt(symbol_energy / 2)
    constellation = amplitude * np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        dtype=complex,
    )
    flat_symbols = np.asarray(equalized_symbols).reshape(-1)
    detected_indices = np.empty(flat_symbols.size, dtype=np.intp)
    batch_size = 65536
    for start in range(0, flat_symbols.size, batch_size):
        batch = flat_symbols[start:start + batch_size]
        distances_squared = np.abs(batch[:, None] - constellation[None, :]) ** 2
        detected_indices[start:start + batch_size] = np.argmin(distances_squared, axis=1)
    return constellation[detected_indices].reshape(equalized_symbols.shape)


def calculate_symbol_error_rate(
    transmitted_symbols: np.ndarray,
    detected_symbols: np.ndarray,
) -> float:
    """Calculate the fraction of transmitted symbols detected incorrectly."""
    if transmitted_symbols.shape != detected_symbols.shape:
        raise ValueError("Transmitted and detected symbol grids must have matching shapes")
    if transmitted_symbols.size == 0:
        raise ValueError("Symbol error rate requires at least one symbol")
    symbol_matches = np.isclose(
        transmitted_symbols,
        detected_symbols,
        rtol=1e-7,
        atol=1e-12,
    )
    return float(np.count_nonzero(~symbol_matches) / transmitted_symbols.size)


def receive_ofdm_symbols(
    received_symbols: np.ndarray,
    channel_response: np.ndarray,
    snr_db: float,
    symbol_energy: float,
    noise_enabled: bool = True,
) -> OFDMReception:
    """Equalize noisy OFDM symbols and make hard QPSK symbol decisions."""
    if received_symbols.shape != channel_response.shape:
        raise ValueError("Received symbols and channel response must have matching shapes")
    noise_variance = (
        estimate_noise_variance(received_symbols, snr_db)
        if noise_enabled
        else 0.0
    )
    equalized_symbols = mmse_equalize(
        received_symbols,
        channel_response,
        noise_variance,
        symbol_energy,
    )
    detected_symbols = detect_qpsk_symbols(equalized_symbols, symbol_energy)
    return OFDMReception(equalized_symbols, detected_symbols, noise_variance)
