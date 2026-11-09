import numpy as np


def create_constellation(modulation: str) -> np.ndarray:
    """Return a unit-average-energy constellation for a supported modulation."""
    match modulation:
        case "BPSK":
            constellation = np.array([-1, 1], dtype=complex)
        case "QPSK":
            constellation = np.array(
                [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                dtype=complex,
            )
        case "16QAM":
            levels = np.array([-3, -1, 1, 3])
            constellation = np.array(
                [in_phase + 1j * quadrature for in_phase in levels for quadrature in levels],
                dtype=complex,
            )
        case "32QAM":
            # Rectangular 4-by-8 32-QAM.
            in_phase_levels = np.array([-3, -1, 1, 3])
            quadrature_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = np.array(
                [
                    in_phase + 1j * quadrature
                    for in_phase in in_phase_levels
                    for quadrature in quadrature_levels
                ],
                dtype=complex,
            )
        case _:
            raise ValueError(
                "modulation must be one of BPSK, QPSK, 16QAM, or 32QAM"
            )

    return constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
