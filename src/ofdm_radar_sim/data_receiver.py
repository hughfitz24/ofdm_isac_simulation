# ==================================
#
# data_receiver.py 
# Defines the behaviour of the receiver that receives the OFDM symbols and performs channel equalisation to estimate the data symbols sent. 
# No radar detection is done here, just standard OFDM processing using an MMSE equaliser. 
#
# ==================================


from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OFDMReception:
    """
    Class that stores the data computed by the OFDM receiver. 

    Stores both the equalised symbols (i.e. after applying channel estimation), and the mapping of those equalised symbols to their nearest Tx symbol
    """
    equalised_symbols: np.ndarray
    symbol_estimations: np.ndarray
    noise_variance: float


def estimate_noise_variance(received_symbols: np.ndarray, snr_db: float) -> float:
    """
    Estimate complex noise power from the received power and configured SNR
    """
    snr_linear = 10 ** (snr_db / 10)
    received_power = float(np.mean(np.abs(received_symbols) ** 2))
    return received_power / (snr_linear + 1)


def mmse_equalise(
    received_symbols: np.ndarray,
    channel_response: np.ndarray,
    noise_variance: float,
    symbol_energy: float,
) -> np.ndarray:
    """
    Apply a MMSE equaliser using known channel gains.
    
    Assumes that the channel gains are known. This is likely to be done using something like pilot symbols. 
    This function assumes that process has been done already, and that the process was perfect. 
    """
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


def ml_receiver(
    equalised_symbols: np.ndarray,
    constellation: np.ndarray,
) -> np.ndarray:
    """
    Perform ML reception on a set of equalised symbols.

    Map equalised symbols to the nearest point in the Tx constellation. 
    """
    if constellation.ndim != 1 or constellation.size == 0:
        raise ValueError("constellation must be a nonempty one-dimensional array")
    flat_symbols = np.asarray(equalised_symbols).reshape(-1)
    detected_indices = np.empty(flat_symbols.size, dtype=np.intp)
    batch_size = 65536
    for start in range(0, flat_symbols.size, batch_size):
        batch = flat_symbols[start:start + batch_size]
        distances_squared = np.abs(batch[:, None] - constellation[None, :]) ** 2
        detected_indices[start:start + batch_size] = np.argmin(distances_squared, axis=1)
    return constellation[detected_indices].reshape(equalised_symbols.shape)


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
    constellation: np.ndarray,
    noise_enabled: bool = True,
) -> OFDMReception:
    """
    Perform MMSE equalisation and ML reception of OFDM symbols. 

    """
    if received_symbols.shape != channel_response.shape:
        raise ValueError("Received symbols and channel response must have matching shapes")
    noise_variance = (
        estimate_noise_variance(received_symbols, snr_db)
        if noise_enabled
        else 0.0
    )
    equalised_symbols = mmse_equalise(
        received_symbols,
        channel_response,
        noise_variance,
        symbol_energy,
    )
    detected_symbols = ml_receiver(equalised_symbols, constellation)
    return OFDMReception(equalised_symbols, detected_symbols, noise_variance)
