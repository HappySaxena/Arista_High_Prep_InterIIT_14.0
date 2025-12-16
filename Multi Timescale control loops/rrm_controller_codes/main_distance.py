#!/usr/bin/env python3
"""
main_distance.py - Compute:
  1) AP–client distances (written back into telemetry JSON)
  2) AP–AP RSSI (saved into a separate JSON file)

Project structure (same folder):

    rrm_slow_loop/
      ├─ main_distance.py
      ├─ distance_calculator.py
      ├─ distance_buffer.py
      ├─ telemetry.json

Outputs:
    telemetry_with_distance.json
    ap_ap_rssi.json
"""

import sys
import os
import json
import math

# -------------------------
# Ensure local project modules are loaded first
# -------------------------
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# -------------------------
# Import DistanceBuffer
# -------------------------
try:
    from distance_buffer import DistanceBuffer
except Exception as e:
    print("ERROR: could not import DistanceBuffer from distance_buffer.py.")
    print("Make sure distance_buffer.py is in the same folder and defines class DistanceBuffer.")
    raise

# -------------------------
# Import distance calculation helpers
# -------------------------
try:
    from distance_calculator import (
        calculate_distance_from_rssi,
        noise_floor_power,
        dbm_to_watt,
        watt_to_dbm,
        wavelength_from_channel,
    )
except Exception as e:
    print("ERROR: could not import distance_calculator functions.")
    print("Make sure distance_calculator.py defines:")
    print("  - calculate_distance_from_rssi(...)")
    print("  - noise_floor_power(lambda_value, NF, T, index)")
    print("  - dbm_to_watt(dbm)")
    print("  - watt_to_dbm(watt)")
    print("  - wavelength_from_channel(channel)")
    raise

# ------------------------------------------------------------------
# Noise figure (NF in dB) per AP, based on chipset
# ------------------------------------------------------------------
AP_NOISE_FIGURES_DB = {
    "ap1": {  # MediaTek MT7921
        "chipset": "MT7921",
        "NF_dB": 4.0,
    },
    "ap2": {  # Realtek RTL8852E
        "chipset": "RTL8852E",
        "NF_dB": 5.0,
    },
    "ap3": {  # Intel AX211
        "chipset": "AX211",
        "NF_dB": 1.5,
    },
}

DEFAULT_NF_DB = 5.0  # fallback if some AP id isn't in the map

# ------------------------------------------------------------------
# Per-link DistanceBuffer store: key = (ap_id, client_mac)
# ------------------------------------------------------------------
link_buffers = {}  # (ap_id, client_mac) -> DistanceBuffer instance

# Per-link "graph distance" store and update threshold
link_graph_distance = {}  # (ap_id, client_mac) -> float
DISTANCE_UPDATE_THRESHOLD = 0.5  # meters; tune as needed

# ------------------------------------------------------------------
# AP–AP distances (meters) - YOU FILL THIS
# ------------------------------------------------------------------
# Use sorted tuple keys so ("ap1", "ap2") == ("ap2", "ap1")
AP_PAIR_DISTANCES_M = {
    # Example values; replace with your real measured distances
    tuple(sorted(["ap1", "ap2"])): 10.0,
    tuple(sorted(["ap1", "ap3"])): 15.0,
    tuple(sorted(["ap2", "ap3"])): 8.0,
}

# ------------------------------------------------------------------
# Interference power black boxes
# ------------------------------------------------------------------
def dummy_client_interference_power(ap_id, client_mac, channel, timestamp):
    """
    Placeholder for AP–client interference estimator.
    Return Pi in Watts.
    """
    # TODO: replace with your real interference calculation
    return 1e-10


def dummy_ap_link_interference_power(tx_ap_id, rx_ap_id, channel, timestamp):
    """
    Placeholder for AP–AP interference estimator.
    Return Pi in Watts.
    """
    # TODO: replace with your real AP–AP interference calculation
    return 1e-10


# ------------------------------------------------------------------
# Core function: update telemetry with AP–client distances
# ------------------------------------------------------------------
def update_telemetry_with_distance(telemetry_batch, get_interference_power):
    """
    telemetry_batch: list[dict]
        List of AP telemetry objects, e.g.:
        [
          {
            "ap_id": "ap1",
            "timestamp": ...,
            "channel": 3,
            "bandwidth": 20,
            "txpower": 3.0,   # dBm
            "num_clients": 2,
            "clients": [
              {"mac": "...", "rssi": -51, ...},
              ...
            ]
          },
          ...
        ]

    get_interference_power: callable
        Pi = get_interference_power(ap_id, client_mac, channel, timestamp)
        (black-box interference estimator from your friend, in Watts)
    """

    # RF constants / assumptions (tune as needed)
    Gi = 2          # tx antenna gain (linear)
    Gr = 2          # rx antenna gain (linear)
    L = 1           # system loss (linear)
    Temp = 290      # temperature (K)
    index = 3       # ΔNF index for noise_floor_power (kept constant)

    for ap in telemetry_batch:
        ap_id = ap["ap_id"]
        channel = ap["channel"]
        ts = ap.get("timestamp", None)

        # --- 1) NF from AP chipset mapping ---
        chip_info = AP_NOISE_FIGURES_DB.get(ap_id)
        if chip_info is not None:
            NF = chip_info["NF_dB"]
            ap["chipset"] = chip_info["chipset"]
            ap["noise_figure_dB"] = NF
        else:
            NF = DEFAULT_NF_DB
            ap["chipset"] = "UNKNOWN"
            ap["noise_figure_dB"] = NF

        # --- 2) Tx power: telemetry txpower is in dBm ---
        Pt_dbm = ap["txpower"]
        Pt_watt = dbm_to_watt(Pt_dbm)

        # --- 3) λ from channel ---
        lambda_value = wavelength_from_channel(channel)

        # --- 4) Noise floor for this AP & channel ---
        Pn = noise_floor_power(lambda_value, NF, Temp, index)

        # --- 5) Process each client attached to this AP ---
        for client in ap.get("clients", []):
            mac = client["mac"]
            rssi_dbm = client["rssi"]

            # RSSI (dBm) -> Watts
            rssi_power = dbm_to_watt(rssi_dbm)

            # Interference power from your friend's module (black box)
            Pi = get_interference_power(ap_id, mac, channel, ts)

            # --- 6) Raw distance from RF link model ---
            try:
                d_raw = calculate_distance_from_rssi(
                    RSSI_power=rssi_power,
                    Pt=Pt_watt,
                    Gi=Gi,
                    Gr=Gr,
                    lambda_value=lambda_value,
                    L=L,
                    Pi=Pi,
                    Pn_floor_power=Pn,
                )
            except ValueError as exc:
                # Too low RSSI vs noise + interference
                client["distance_raw_m"] = None
                client["distance_mean_m"] = None
                client["distance_std_m"] = None
                client["distance_adj_m"] = None
                client["distance_graph_m"] = None
                client["distance_error"] = str(exc)
                continue

            # --- 7) Smooth / adjust distance using DistanceBuffer per link ---
            key = (ap_id, mac)
            buf = link_buffers.setdefault(
                key,
                DistanceBuffer(max_size=100, min_variance_threshold=5.0),
            )

            stats = buf.add_distance(d_raw, rssi=rssi_dbm)

            # --- 8) Decide "graph distance" with thresholded updates ---
            # Candidate distance: use adjusted_distance if available, else raw
            candidate = stats["adjusted_distance"]
            if candidate is None:
                candidate = d_raw

            prev_graph = link_graph_distance.get(key)

            if prev_graph is None:
                # First ever value for this link: use candidate immediately
                graph_distance = candidate
            else:
                if candidate is None:
                    graph_distance = prev_graph
                else:
                    if abs(candidate - prev_graph) >= DISTANCE_UPDATE_THRESHOLD:
                        graph_distance = candidate
                    else:
                        graph_distance = prev_graph

            # Store back into the per-link graph-distance cache
            link_graph_distance[key] = graph_distance

            # --- 9) Write results back into telemetry ---
            client["distance_raw_m"] = d_raw
            client["distance_mean_m"] = stats["mean"]
            client["distance_std_m"] = stats["std"]
            client["distance_adj_m"] = stats["adjusted_distance"]
            client["distance_reset"] = stats["reset"]
            client["distance_buffer_len"] = stats["buffer_length"]

            # Field your interference graph should use:
            client["distance_graph_m"] = graph_distance

    return telemetry_batch


# ------------------------------------------------------------------
# AP–AP RSSI computation (only if same channel)
# ------------------------------------------------------------------
def compute_ap_to_ap_rssi(telemetry_batch, ap_pair_distances, get_ap_interference_power):
    """
    Compute RSSI between APs, assuming constant distances between APs.

    - Only consider AP pairs that share the SAME channel.
    - Distances are taken from ap_pair_distances (meters).
    - Interference power is provided by get_ap_interference_power.

    Returns:
        list[dict] of links, one entry per direction (tx -> rx)
    """

    # RF constants (same as above)
    Gi = 2
    Gr = 2
    L = 1
    Temp = 290
    index = 3  # ΔNF index for noise_floor_power

    # Map AP id -> telemetry dict
    ap_map = {ap["ap_id"]: ap for ap in telemetry_batch}
    ap_ids = list(ap_map.keys())

    links = []

    # Iterate unordered pairs (i < j)
    for i in range(len(ap_ids)):
        for j in range(i + 1, len(ap_ids)):
            ap_id_1 = ap_ids[i]
            ap_id_2 = ap_ids[j]

            ap1 = ap_map[ap_id_1]
            ap2 = ap_map[ap_id_2]

            ch1 = ap1["channel"]
            ch2 = ap2["channel"]

            # Only if they are on the same channel
            if ch1 != ch2:
                continue

            channel = ch1  # common channel

            # Look up distance (symmetric)
            dist_key = tuple(sorted([ap_id_1, ap_id_2]))
            d = ap_pair_distances.get(dist_key)
            if d is None:
                # No distance known for this pair -> skip
                continue

            # Use some representative timestamp (min of the two, for example)
            ts = min(ap1.get("timestamp", 0), ap2.get("timestamp", 0))

            # Compute λ
            lambda_value = wavelength_from_channel(channel)

            # --- Direction 1: ap1 -> ap2 ---
            link_1 = _compute_single_ap_link(
                tx_ap=ap1,
                rx_ap=ap2,
                distance_m=d,
                channel=channel,
                lambda_value=lambda_value,
                Temp=Temp,
                index=index,
                get_ap_interference_power=get_ap_interference_power,
                timestamp=ts,
            )
            if link_1 is not None:
                links.append(link_1)

            # --- Direction 2: ap2 -> ap1 ---
            link_2 = _compute_single_ap_link(
                tx_ap=ap2,
                rx_ap=ap1,
                distance_m=d,
                channel=channel,
                lambda_value=lambda_value,
                Temp=Temp,
                index=index,
                get_ap_interference_power=get_ap_interference_power,
                timestamp=ts,
            )
            if link_2 is not None:
                links.append(link_2)

    return links


def _compute_single_ap_link(
    tx_ap,
    rx_ap,
    distance_m,
    channel,
    lambda_value,
    Temp,
    index,
    get_ap_interference_power,
    timestamp,
):
    """
    Compute RSSI (in dBm) for a single AP->AP direction.

    Uses:
        RSSI_power = (Pt * Gi * Gr * λ²) / ((4π)² * d² * L) + Pn + Pi
    """

    Gi = 2
    Gr = 2
    L = 1

    tx_id = tx_ap["ap_id"]
    rx_id = rx_ap["ap_id"]

    # Tx power
    Pt_dbm = tx_ap["txpower"]
    Pt_watt = dbm_to_watt(Pt_dbm)

    # Rx NF
    chip_info = AP_NOISE_FIGURES_DB.get(rx_id)
    if chip_info is not None:
        NF_rx = chip_info["NF_dB"]
    else:
        NF_rx = DEFAULT_NF_DB

    # Noise floor at receiver
    Pn = noise_floor_power(lambda_value, NF_rx, Temp, index)

    # Interference power from black box
    Pi = get_ap_interference_power(tx_id, rx_id, channel, timestamp)

    # Useful signal power from Friis
    numerator = Pt_watt * Gi * Gr * (lambda_value ** 2)
    denominator = (4 * math.pi) ** 2 * (distance_m ** 2) * L
    useful_signal = numerator / denominator

    # Total RSSI power
    RSSI_power = useful_signal + Pn + Pi

    if RSSI_power <= 0:
        # Should not happen, but guard
        return None

    rssi_dbm = watt_to_dbm(RSSI_power)

    return {
        "tx_ap": tx_id,
        "rx_ap": rx_id,
        "channel": channel,
        "distance_m": distance_m,
        "Pt_dbm_tx": Pt_dbm,
        "NF_rx_dB": NF_rx,
        "Pi_W": Pi,
        "RSSI_dBm": rssi_dbm,
        "timestamp": timestamp,
    }


# ------------------------------------------------------------------
# Helpers to load/save telemetry JSON from same directory
# ------------------------------------------------------------------
def load_telemetry_json(filename="telemetry_full.json"):
    """Load telemetry JSON from the same directory as this script."""
    path = os.path.join(project_path, filename)
    with open(path, "r") as f:
        data = json.load(f)

    # Allow either a top-level list or {"aps": [...]}
    if isinstance(data, dict) and "aps" in data:
        return data["aps"], True, path
    elif isinstance(data, list):
        return data, False, path
    else:
        raise ValueError("Unsupported telemetry JSON structure.")


def save_telemetry_json(telemetry_batch, original_was_dict, original_path,
                        out_filename="telemetry_with_distance.json"):
    """Save updated telemetry JSON alongside the original."""
    out_path = os.path.join(project_path, out_filename)

    if original_was_dict:
        # reconstruct {"aps": [...]}
        with open(original_path, "r") as f:
            base = json.load(f)
        base["aps"] = telemetry_batch
        out_data = base
    else:
        out_data = telemetry_batch

    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    return out_path


def save_ap_ap_rssi_json(links, out_filename="ap_ap_rssi.json"):
    """Save AP–AP RSSI data to a separate JSON file."""
    out_path = os.path.join(project_path, out_filename)
    with open(out_path, "w") as f:
        json.dump(links, f, indent=2)
    return out_path


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load telemetry.json from this directory
    telemetry, original_is_dict, original_path = load_telemetry_json(
        filename="telemetry_full.json"
    )

    # 2) Update telemetry with AP–client distances
    updated_telemetry = update_telemetry_with_distance(
        telemetry_batch=telemetry,
        get_interference_power=dummy_client_interference_power,  # replace with real Pi
    )

    # 3) Save AP–client distances back into telemetry_with_distance.json
    out_file = save_telemetry_json(
        telemetry_batch=updated_telemetry,
        original_was_dict=original_is_dict,
        original_path=original_path,
        out_filename="telemetry_with_distance.json",
    )

    print(f"\nUpdated telemetry written to: {out_file}\n")

    # 4) Compute AP–AP RSSI (only same-channel AP pairs)
    ap_links = compute_ap_to_ap_rssi(
        telemetry_batch=updated_telemetry,
        ap_pair_distances=AP_PAIR_DISTANCES_M,
        get_ap_interference_power=dummy_ap_link_interference_power,
    )

    # 5) Save AP–AP RSSI to separate JSON
    ap_rssi_file = save_ap_ap_rssi_json(ap_links, out_filename="ap_ap_rssi.json")
    print(f"AP–AP RSSI written to: {ap_rssi_file}\n")

    # 6) Print a short summary
    print("=== AP–client distances (graph distance) ===")
    for ap in updated_telemetry:
        print(f"AP {ap['ap_id']} ({ap.get('chipset', 'UNKNOWN')}) channel {ap['channel']}:")
        for c in ap.get("clients", []):
            print(
                f"  Client {c['mac']}: RSSI={c['rssi']} dBm, "
                f"dist_graph={c.get('distance_graph_m')}"
            )

    print("\n=== AP–AP RSSI (same-channel pairs only) ===")
    for link in ap_links:
        print(
            f"  {link['tx_ap']} -> {link['rx_ap']} | "
            f"ch={link['channel']} | d={link['distance_m']} m | "
            f"RSSI={link['RSSI_dBm']:.2f} dBm"
        )
