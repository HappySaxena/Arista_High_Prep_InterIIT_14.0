import math

SPEED_OF_LIGHT = 3e8  # m/s

# ================================================================
#  UTILITY FUNCTIONS: Power Conversions
# ================================================================

def dbm_to_watt(dbm):
    """
    Convert power from dBm to Watt.
    Formula: P(W) = 10^((P(dBm) - 30) / 10)
    """
    return 10 ** ((dbm - 30) / 10)


def watt_to_dbm(watt):
    """
    Convert power from Watt to dBm.
    Formula: P(dBm) = 10*log10(P(W)) + 30
    """
    return 10 * math.log10(watt) + 30


# ================================================================
#  CHANNEL → FREQUENCY / WAVELENGTH HELPERS
# ================================================================

def channel_to_frequency_hz(channel: int) -> float:
    """
    Map IEEE 802.11 Wi-Fi channel to center frequency in Hz.

    - 2.4 GHz band (ch 1–14): f(MHz) = 2407 + 5*ch
    - 5 GHz band (DFS + non-DFS): f(MHz) = 5000 + 5*ch
      (e.g., ch36 → 5180 MHz, ch52 → 5260 MHz, ch100 → 5500 MHz)
    """
    if channel < 1:
        raise ValueError(f"Unsupported Wi-Fi channel: {channel}")

    if 1 <= channel <= 14:
        # 2.4 GHz
        freq_mhz = 2407 + 5 * channel
    else:
        # 5 GHz band (DFS + non-DFS)
        freq_mhz = 5000 + 5 * channel

    return freq_mhz * 1e6  # Hz


def wavelength_from_channel(channel: int) -> float:
    """
    Return λ (meters) for a given Wi-Fi channel.
    """
    f_hz = channel_to_frequency_hz(channel)
    return SPEED_OF_LIGHT / f_hz


# ================================================================
#  NOISE FLOOR CALCULATION (in POWER)
# ================================================================

def noise_floor_power(lambda_value, NF, T, index):
    """
    Calculate noise floor in WATT using the full formula:

    N_floor(dBm) = -174 + 10log10(B) + NF + 10log10(T/290) + ΔNF

    IMPORTANT:
    - B (bandwidth) is calculated from the wavelength: B = c / lambda
    - ΔNF depends on 'index':
          index=1 → +2 dB
          index=2 → +3 dB
          index=3 → +1.5 dB
    """

    # ----- speed of light -----
    c = SPEED_OF_LIGHT

    # ----- calculate bandwidth from lambda -----
    B = c / lambda_value

    # ----- choose ΔNF -----
    if index == 1:
        deltaNF = 2        # dB
    elif index == 2:
        deltaNF = 3        # dB
    elif index == 3:
        deltaNF = 1.5      # dB
    else:
        raise ValueError("Index must be 1, 2, or 3.")

    # ----- calculate noise floor in dBm -----
    N_floor_dBm = (
        -174
        + 10 * math.log10(B)
        + NF
        + 10 * math.log10(T / 290)
        + deltaNF
    )

    # ----- convert noise floor to power in Watts -----
    return dbm_to_watt(N_floor_dBm)


# ================================================================
#  DISTANCE CALCULATION FROM RSSI POWER
# ================================================================

def calculate_distance_from_rssi(
    RSSI_power, Pt, Gi, Gr, lambda_value, L, Pi, Pn_floor_power
):
    """
    Calculate distance 'd' using the EXACT formula:

    RSSI = (Pt*Gi*Gr*λ²) / ((4π)² * d² * L) + Pn_floor + Pi

    Rearranged for distance:
    
    d = sqrt( (Pt*Gi*Gr*λ²) / ( (4π)² * L * (RSSI - Pn_floor - Pi) ) )
    """

    # ----- noise + interference -----
    noise_interference = Pn_floor_power + Pi

    # ----- useful signal must be positive -----
    useful_signal = RSSI_power - noise_interference

    if useful_signal <= 0:
        raise ValueError(
            "Error: RSSI power too low. (RSSI - Noise - Interference <= 0)"
        )

    numerator = Pt * Gi * Gr * (lambda_value ** 2)
    denominator = useful_signal * L * (4 * math.pi) ** 2

    # ----- final distance -----
    d = math.sqrt(numerator / denominator)
    return d


# ================================================================
#  OPTIONAL EXAMPLE USAGE
# ================================================================

if __name__ == "__main__":

    RSSI_power = 0.0000001     # received signal in Watts
    Pt = 0.1               # transmit power = 0.1W
    Gi = 2                 # Tx antenna gain (linear)
    Gr = 2                 # Rx antenna gain (linear)
    lambda_value = 0.125   # 2.4GHz approx = 0.125m
    L = 1                  # system loss
    Pi = 1e-10             # interference power
    NF = 5                 # noise figure (dB)
    Temp = 290             # temperature in Kelvin
    index = 3              # ΔNF = 1.5 dB

    Pn_floor = noise_floor_power(lambda_value, NF, Temp, index)

    distance = calculate_distance_from_rssi(
        RSSI_power, Pt, Gi, Gr, lambda_value, L, Pi, Pn_floor
    )

    print(f"Estimated Distance = {distance:.4f} meters")
