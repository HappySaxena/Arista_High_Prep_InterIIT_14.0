# fast_loop_runtime.py
# --------------------
# Fast-loop logic for DFS + interference-driven channel / width changes.
# Integrates with your live telemetry (telemetry_buffer.snapshot()) and
# uses per-channel interference_power_dBm from channel_summary.
#
# Key behaviors:
#   - 2.4 GHz: only 20 MHz, no BW increase.
#   - 5 GHz 40 MHz:
#       * If secondary channel bad -> narrow to 20 MHz (stay on primary).
#       * If primary (or both) bad -> leave channel, pick best 20 MHz.
#   - DFS channels (DFS_CHANNELS set): if interference exceeds threshold,
#     treat as radar and move to non-DFS 20 MHz.
#
# Proposals are returned as a list of dicts:
#   { "ap_id": str, "type": "channel" | "width", "value": int, "reason": str }

from typing import Dict, List
from collections import defaultdict
import time

# =========================
# FAST-LOOP ROBUSTNESS KNOBS
# =========================

# At most N APs changed per fast-loop tick
MAX_FASTLOOP_CHANGES_PER_STEP = 1

# Hard cap on how many APs we want on the same channel (from fast-loop’s view)
MAX_APS_PER_CHANNEL_FASTLOOP = 1

# How many summary snapshots per AP before we trust interference dBm for moves
FASTLOOP_WARMUP_SUMMARIES = 3          # e.g. need 3 summary pushes per AP
FASTLOOP_DEBUG = True

# Only move if new channel looks at least this many dB better
MIN_INTERFERENCE_IMPROVEMENT_DB = 6.0  # require >=6 dB improvement

# Don’t churn APs with no clients
MIN_CLIENTS_TO_MOVE = 1

# Ignore fast loop for first X seconds (let RL / configs settle)
FASTLOOP_STARTUP_GRACE_SEC = 120.0
_fastloop_start_ts = time.time()

# Minimum time between fast-loop changes per AP (seconds)
FASTLOOP_MIN_CHANGE_INTERVAL_SEC = 600.0   # 10 minutes

# =========================
# CHANNEL LISTS
# =========================

# 2.4 GHz channels. IMPORTANT: we do NOT allow 12 and 13.
CHANNELS_24GHZ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 5 GHz channels (same list as in your controller).
CHANNELS_5GHZ = [36, 40, 44, 48, 149]

ALL_CHANNELS = CHANNELS_24GHZ + CHANNELS_5GHZ

# DFS channels you want to treat as "avoid unless desperate".
DFS_CHANNELS = {36, 40}  # adjust if you mark more channels as DFS

# =========================
# WEIGHTS / THRESHOLDS
# =========================

# "More score = worse"
WEIGHT_AP_COUNT = 20.0           # cost per AP already on candidate channel
WEIGHT_CLIENT_COUNT = 5.0        # cost per client already on candidate channel
WEIGHT_INTERFERENCE = 150.0      # penalty if candidate channel has severe interference
WEIGHT_DFS = 10000.0             # avoid DFS unless absolutely forced
WEIGHT_OVERLAP = 12.0            # 2.4 GHz overlap penalty
WEIGHT_CHANNEL_STABILITY = 3.0   # small bonus for staying on current channel

NON_OVERLAP_SPACING_24GHZ = 5    # 1 & 6 are non-overlapping at 20 MHz

# Interference thresholds (dBm)
# - channel_summary gives interference_power_dBm. -100 is "no data".
# - Closer to 0 or positive => very strong interference.
INTERFERENCE_SEVERE_DBM = -10.0        # generic "this channel is bad" threshold
DFS_INTERFERENCE_SEVERE_DBM = -40.0    # slightly more sensitive for DFS radar workaround

# "Desperate DFS" logic
ALLOW_DFS_IF_DESPERATE = True
DESPERATE_INTERFERENCE_CLIENTS_THRESHOLD = 5
DESPERATE_SCORE_DIFF_THRESHOLD = 50.0

# Fallback if channel_summary missing
RETRY_THRESHOLD = 50
TX_FAILED_THRESHOLD = 0

# =========================
# INTERNAL STATE
# =========================

# How many summaries we’ve seen per AP
_fastloop_summary_seen: Dict[str, int] = defaultdict(int)

# Last time we changed this AP via fast loop
_fastloop_last_change_ts: Dict[str, float] = defaultdict(float)

# Last *planned* target channel per AP (to handle telemetry lag)
_fastloop_last_target_channel: Dict[str, int] = {}

# =========================
# CHANNEL-SUMMARY INTERFERENCE HELPERS
# =========================

def get_channel_interference_dbm(ap_telemetry: Dict, channel: int | None = None) -> float | None:
    """
    Look up interference_power_dBm for a given 20 MHz channel from AP's channel_summary.

    Returns:
      - float(dBm) if valid and not clearly "no data"
      - None if no info or sentinel (<= -95 dBm)
    """
    if channel is None:
        channel = ap_telemetry.get("channel")
    if channel is None:
        return None

    summary = ap_telemetry.get("channel_summary") or ap_telemetry.get("summary") or {}
    ch_key = f"ch{int(channel)}"
    ch_info = summary.get(ch_key)
    if not ch_info:
        return None

    ip_dbm = ch_info.get("interference_power_dBm")
    if ip_dbm is None:
        return None

    try:
        ip_dbm = float(ip_dbm)
    except (TypeError, ValueError):
        return None

    # In your examples, -100 is sentinel for "no interference / no data".
    if ip_dbm <= -95.0:
        return None

    return ip_dbm


def fallback_interference_by_counters(ap_telemetry: Dict) -> bool:
    """
    Fallback: if no channel_summary, decide interference by tx_failed/tx_retries.
    """
    for c in ap_telemetry.get("clients", []):
        if c.get("tx_failed", 0) > TX_FAILED_THRESHOLD:
            return True
        if c.get("tx_retries", 0) >= RETRY_THRESHOLD:
            return True
    return False


# =========================
# OVERLAP + SCORING
# =========================

def channels_overlap_24ghz(ch1: int, ch2: int, bandwidth: int = 20) -> bool:
    """
    True if two 2.4 GHz channels overlap at 20 MHz.
    """
    if ch1 == ch2:
        return True
    if ch1 in CHANNELS_24GHZ and ch2 in CHANNELS_24GHZ:
        return abs(ch1 - ch2) < NON_OVERLAP_SPACING_24GHZ
    # Cross-band: treat 2.4 vs 5 GHz as non-overlapping
    return False


def get_clients_count_on_channel(candidate_channel: int,
                                 telemetry_map: Dict[str, Dict]) -> int:
    total = 0
    for _, tel in telemetry_map.items():
        if tel.get("channel") == candidate_channel:
            total += int(tel.get("num_clients", 0))
    return total


def score_channel(candidate_channel: int,
                  ap_id: str,
                  telemetry_map: Dict[str, Dict],
                  channel_occupancy: Dict[int, int] | None = None,
                  return_details: bool = False):
    """
    Compute numeric score for candidate_channel on AP ap_id.
    Lower is better.

    If return_details=True, returns (score, details_dict).
    Otherwise returns score as float.
    """
    if candidate_channel not in ALL_CHANNELS:
        return (float("inf"), {}) if return_details else float("inf")

    ap_tel = telemetry_map.get(ap_id, {})
    current_ch = ap_tel.get("channel")
    score = 0.0

    # ---- 0) AP stacking penalty ----
    if channel_occupancy is not None:
        aps_on = channel_occupancy.get(candidate_channel, 0)
        # Don't count this AP against itself if it is already on that channel
        if current_ch == candidate_channel:
            aps_on = max(0, aps_on - 1)
    else:
        aps_on = 0
        for _, tel in telemetry_map.items():
            ch = int(tel.get("channel", 0) or 0)
            if ch == candidate_channel:
                aps_on += 1
        if current_ch == candidate_channel:
            aps_on = max(0, aps_on - 1)

    score += WEIGHT_AP_COUNT * aps_on

    # ---- 1) DFS penalty ----
    is_dfs = candidate_channel in DFS_CHANNELS
    if is_dfs:
        score += WEIGHT_DFS

    # ---- 2) client count penalty ----
    clients_on = get_clients_count_on_channel(candidate_channel, telemetry_map)
    score += WEIGHT_CLIENT_COUNT * clients_on

    # ---- 3) interference penalty ----
    ip_dbm_candidate = get_channel_interference_dbm(ap_tel, candidate_channel)
    if ip_dbm_candidate is not None and ip_dbm_candidate >= INTERFERENCE_SEVERE_DBM:
        score += WEIGHT_INTERFERENCE

    # ---- 4) 2.4 GHz overlap penalty ----
    overlap_count = 0
    for other_ap, tel in telemetry_map.items():
        if other_ap == ap_id:
            continue
        other_ch = tel.get("channel")
        if other_ch is None:
            continue
        if channels_overlap_24ghz(candidate_channel, int(other_ch)):
            overlap_count += 1
    score += WEIGHT_OVERLAP * overlap_count

    # ---- 5) stability bonus ----
    stability_bonus = False
    if current_ch == candidate_channel:
        score -= WEIGHT_CHANNEL_STABILITY
        stability_bonus = True

    if not return_details:
        return score

    details = {
        "aps_on": aps_on,
        "clients_on": clients_on,
        "ip_dbm": ip_dbm_candidate,
        "overlap_count": overlap_count,
        "is_dfs": is_dfs,
        "stability_bonus": stability_bonus,
    }
    return score, details


# =========================
# CHANNEL SELECTION & DFS
# =========================

def choose_best_channel_for_ap(ap_id: str,
                               telemetry_map: Dict[str, Dict],
                               channel_occupancy: Dict[int, int] | None = None,
                               debug: bool = False) -> Dict:
    """
    Evaluate all candidate channels and return:
      {
        "best_channel": int,
        "best_score": float,
        "best_non_dfs_channel": int or None,
        "best_non_dfs_score": float
      }

    Behavior:
      - If channel_occupancy is provided, we first try channels that currently
        have < MAX_APS_PER_CHANNEL_FASTLOOP APs.
      - If *all* channels are "full" by that rule, we fall back to ALL_CHANNELS.
      - If debug=True, we print a table of (channel, score, details).
    """
    ap_tel = telemetry_map.get(ap_id, {})
    current_ch = ap_tel.get("channel")

    if channel_occupancy is not None:
        preferred = [
            ch for ch in ALL_CHANNELS
            if channel_occupancy.get(ch, 0) < MAX_APS_PER_CHANNEL_FASTLOOP
        ]
        candidate_list = preferred if preferred else ALL_CHANNELS
    else:
        candidate_list = ALL_CHANNELS

    best = None
    best_score = float("inf")
    best_non_dfs = None
    best_non_dfs_score = float("inf")

    debug_rows = []

    for ch in candidate_list:
        if debug:
            sc, details = score_channel(
                ch, ap_id, telemetry_map,
                channel_occupancy=channel_occupancy,
                return_details=True,
            )
            debug_rows.append((ch, sc, details))
        else:
            sc = score_channel(
                ch, ap_id, telemetry_map,
                channel_occupancy=channel_occupancy,
                return_details=False,
            )

        if sc < best_score:
            best_score = sc
            best = ch
        if ch not in DFS_CHANNELS and sc < best_non_dfs_score:
            best_non_dfs_score = sc
            best_non_dfs = ch

    if debug and debug_rows:
        print(f"\n[FAST-LOOP-DEBUG] Channel scores for ap={ap_id}, current_ch={current_ch}")
        print("  ch | score    | aps_on | clients_on | ip_dbm   | overlap | dfs | stay_bonus")
        print("  ---+----------+--------+------------+----------+---------+-----+-----------")
        for ch, sc, d in sorted(debug_rows, key=lambda x: x[1]):
            ip = d["ip_dbm"]
            ip_str = f"{ip:7.1f}" if ip is not None else "   None"
            print(
                f"  {ch:2d} | {sc:8.2f} |"
                f" {d['aps_on']:6d} |"
                f" {d['clients_on']:10d} |"
                f" {ip_str} |"
                f" {d['overlap_count']:7d} |"
                f" {'Y' if d['is_dfs'] else 'N'}   |"
                f" {'Y' if d['stability_bonus'] else 'N'}"
            )
        print(
            f"[FAST-LOOP-DEBUG] ap={ap_id}: best={best} (score={best_score:.2f}), "
            f"best_non_dfs={best_non_dfs} (score={best_non_dfs_score:.2f})\n"
        )

    return {
        "best_channel": best,
        "best_score": best_score,
        "best_non_dfs_channel": best_non_dfs,
        "best_non_dfs_score": best_non_dfs_score,
    }


def count_suffering_clients(ap_telemetry: Dict) -> int:
    suffering = 0
    for c in ap_telemetry.get("clients", []):
        if c.get("tx_failed", 0) > TX_FAILED_THRESHOLD or c.get("tx_retries", 0) >= RETRY_THRESHOLD:
            suffering += 1
    return suffering


def should_allow_dfs_as_last_resort(ap_telemetry: Dict,
                                    telemetry_map: Dict[str, Dict],
                                    channel_occupancy: Dict[int, int] | None = None) -> bool:
    """
    Decide if we can use a DFS channel as last-resort:
      - ALLOW_DFS_IF_DESPERATE must be True
      - Enough suffering clients
      - Best non-DFS is much worse than best overall
      - (optionally) use occupancy-aware scoring if channel_occupancy is provided
    """
    if not ALLOW_DFS_IF_DESPERATE:
        return False

    suffering = count_suffering_clients(ap_telemetry)
    if suffering < DESPERATE_INTERFERENCE_CLIENTS_THRESHOLD:
        return False

    ap_id = ap_telemetry.get("ap_id")
    if not ap_id:
        return False

    chinfo = choose_best_channel_for_ap(
        ap_id,
        telemetry_map,
        channel_occupancy=channel_occupancy,
        debug=FASTLOOP_DEBUG,
    )

    best_all = chinfo["best_score"]
    best_non_dfs = (
        chinfo["best_non_dfs_score"]
        if chinfo["best_non_dfs_score"] is not None
        else float("inf")
    )

    if (best_non_dfs - best_all) > DESPERATE_SCORE_DIFF_THRESHOLD:
        return True
    return False


# =========================
# INTERFERENCE CLASSIFICATION
# =========================

def classify_5ghz_40mhz_interference(ap_tel: Dict,
                                     thresh_dbm: float = INTERFERENCE_SEVERE_DBM) -> str:
    """
    For a 5 GHz, 40 MHz AP, classify interference as:
      "none", "primary_only", "secondary_only", "both"

    Assumes primary = ap_tel["channel"], secondary = primary + 4
    """
    primary_ch = int(ap_tel.get("channel", 0) or 0)
    if primary_ch <= 14:
        return "none"  # not 5 GHz

    bw = int(ap_tel.get("bandwidth", 20) or 20)
    if bw <= 20:
        return "none"

    primary_ip = get_channel_interference_dbm(ap_tel, primary_ch)
    secondary_ip = get_channel_interference_dbm(ap_tel, primary_ch + 4)

    primary_bad = primary_ip is not None and primary_ip >= thresh_dbm
    secondary_bad = secondary_ip is not None and secondary_ip >= thresh_dbm

    if primary_bad and secondary_bad:
        return "both"
    if primary_bad:
        return "primary_only"
    if secondary_bad:
        return "secondary_only"
    return "none"


def is_dfs_radar_like_event(ap_tel: Dict) -> bool:
    """
    DFS workaround:
      - If AP is on a DFS channel and interference on that DFS channel exceeds
        DFS_INTERFERENCE_SEVERE_DBM, treat this as "radar detected".
      - If orchestrator sets ap_tel["dfs"] True, we also treat that as radar.
    """
    ch = int(ap_tel.get("channel", 0) or 0)
    dfs_flag = bool(ap_tel.get("dfs", False))

    if ch in DFS_CHANNELS:
        # Real dfs flag OR high interference on DFS primary channel
        ip_dbm = get_channel_interference_dbm(ap_tel, ch)
        if dfs_flag:
            return True
        if ip_dbm is not None and ip_dbm >= DFS_INTERFERENCE_SEVERE_DBM:
            return True

    return False


# =========================
# MAIN FAST LOOP ENTRYPOINT
# =========================

def generate_fastloop_proposals_from_snapshot(snapshot: Dict[str, Dict]) -> List[Dict]:
    """
    Use latest snapshot = telemetry_buffer.snapshot() and generate fast-loop proposals.

    Returns: list of dicts:
      { "ap_id": str, "type": "channel" | "width", "value": int, "reason": str }
    """
    proposals: List[Dict] = []
    telemetry_map = snapshot
    now = time.time()

    # Global startup grace: let RL settle, don't fast-loop in first X seconds
    if now - _fastloop_start_ts < FASTLOOP_STARTUP_GRACE_SEC:
        return []

    # ---- current channel occupancy, including last planned target ----
    channel_occupancy: Dict[int, int] = defaultdict(int)
    for ap_id, tel in telemetry_map.items():
        # If we had a planned target for this AP, prefer that (telemetry may lag)
        planned = _fastloop_last_target_channel.get(ap_id)
        ch_snapshot = int(tel.get("channel", 0) or 0)
        ch = planned if planned is not None else ch_snapshot
        if ch:
            channel_occupancy[ch] += 1

    # ---- main loop over APs ----
    for ap_id, raw_tel in telemetry_map.items():
        # Shallow copy so we can add ap_id and not mutate original snapshot
        tel = dict(raw_tel)
        tel["ap_id"] = ap_id

        cur_channel = int(tel.get("channel", 0) or 0)
        cur_bw = int(tel.get("bandwidth", 20) or 20)

        is_24ghz = cur_channel in CHANNELS_24GHZ
        is_5ghz = cur_channel in CHANNELS_5GHZ

        # ---- RATE LIMIT: don't change this AP too often via fast loop ----
        last_change = _fastloop_last_change_ts[ap_id]
        if last_change > 0.0 and (now - last_change) < FASTLOOP_MIN_CHANGE_INTERVAL_SEC:
            # Leave this AP to slow loop / RL for now
            continue

        # ---- SUMMARY WARM-UP: only trust interference after seeing it a few times ----
        if tel.get("channel_summary") or tel.get("summary"):
            _fastloop_summary_seen[ap_id] += 1
        summaries_seen = _fastloop_summary_seen[ap_id]

        # --- 1) DFS radar-like event => MOVE OFF DFS TO NON-DFS 20 MHz ---
        if is_dfs_radar_like_event(tel):
            chinfo = choose_best_channel_for_ap(ap_id, telemetry_map, channel_occupancy, debug=FASTLOOP_DEBUG)
            target_non_dfs = chinfo["best_non_dfs_channel"]

            # Prefer best non-DFS; if none, allow DFS best only if "desperate"
            if target_non_dfs is None:
                if should_allow_dfs_as_last_resort(tel, telemetry_map, channel_occupancy):
                    target = chinfo["best_channel"]
                else:
                    # fallback: any non-DFS different from current; if none, stay
                    non_dfs = [c for c in ALL_CHANNELS if c not in DFS_CHANNELS and c != cur_channel]
                    target = non_dfs[0] if non_dfs else cur_channel
            else:
                target = target_non_dfs

            if target is not None and target != cur_channel:
                proposals.append({
                    "ap_id": ap_id,
                    "type": "channel",
                    "value": int(target),
                    "reason": "fastloop_dfs_evict",
                })
                _fastloop_last_change_ts[ap_id] = now
                _fastloop_last_target_channel[ap_id] = int(target)

                # update planned occupancy
                if cur_channel:
                    channel_occupancy[cur_channel] = max(0, channel_occupancy[cur_channel] - 1)
                channel_occupancy[target] += 1

                if len(proposals) >= MAX_FASTLOOP_CHANGES_PER_STEP:
                    break
            # DFS handled; next AP
            continue

        # --- 2) Non-DFS: handle interference based on band + bandwidth ---

        # 2.1: 5 GHz, 40 MHz -> primary / secondary logic
        if is_5ghz and cur_bw > 20:
            # Don't react to a single noisy summary; wait for a few
            if summaries_seen >= FASTLOOP_WARMUP_SUMMARIES:
                cls = classify_5ghz_40mhz_interference(tel, INTERFERENCE_SEVERE_DBM)
            else:
                cls = "none"

            if cls == "secondary_only":
                proposals.append({
                    "ap_id": ap_id,
                    "type": "width",
                    "value": 20,
                    "reason": "fastloop_width_tighten_secondary_only_interference",
                })
                _fastloop_last_change_ts[ap_id] = now
                if len(proposals) >= MAX_FASTLOOP_CHANGES_PER_STEP:
                    break
                continue

            elif cls in ("primary_only", "both"):
                chinfo = choose_best_channel_for_ap(ap_id, telemetry_map, channel_occupancy, debug=FASTLOOP_DEBUG)
                target = chinfo["best_non_dfs_channel"] or chinfo["best_channel"]

                if target is not None and target != cur_channel:
                    proposals.append({
                        "ap_id": ap_id,
                        "type": "channel",
                        "value": int(target),
                        "reason": "fastloop_channel_change_primary_or_both_interference",
                    })
                    _fastloop_last_change_ts[ap_id] = now
                    _fastloop_last_target_channel[ap_id] = int(target)

                    if cur_channel:
                        channel_occupancy[cur_channel] = max(0, channel_occupancy[cur_channel] - 1)
                    channel_occupancy[target] += 1

                    if len(proposals) >= MAX_FASTLOOP_CHANGES_PER_STEP:
                        break
                continue

            else:
                # cls == "none" -> no severe interference across 40 MHz; fast loop quiet
                continue

        # 2.2: 5 GHz, 20 MHz OR 2.4 GHz, 20 MHz
        num_clients = int(tel.get("num_clients", 0))

        ip_dbm_current = get_channel_interference_dbm(tel, cur_channel)

        # Decide if current channel is "interfered"
        if ip_dbm_current is None:
            # No valid summary for this channel -> only rely on counters
            interfered = fallback_interference_by_counters(tel)
        else:
            # We have a numeric summary but only trust it after warm-up
            if summaries_seen < FASTLOOP_WARMUP_SUMMARIES:
                interfered = False
            else:
                interfered = ip_dbm_current >= INTERFERENCE_SEVERE_DBM

        # Don’t move APs with no clients, or if channel not considered bad
        if not interfered or num_clients < MIN_CLIENTS_TO_MOVE:
            continue

        # We think current channel is bad -> see if there is a meaningfully better one
        chinfo = choose_best_channel_for_ap(ap_id, telemetry_map, channel_occupancy, debug=FASTLOOP_DEBUG)
        target = chinfo["best_non_dfs_channel"] or chinfo["best_channel"]

        if target is None or target == cur_channel:
            continue

        # If we **do** have numeric dBm for current channel, enforce minimum
        # dB improvement before moving. If we only had counters (ip_dbm_current
        # is None), skip this check.
        if ip_dbm_current is not None:
            ip_dbm_target = get_channel_interference_dbm(tel, target)
            if ip_dbm_target is None:
                # Treat "no data" on candidate as very quiet
                ip_dbm_target = -100.0

            improvement = ip_dbm_current - ip_dbm_target
            if FASTLOOP_DEBUG:
                print(
                    f"[FAST-LOOP-DEBUG] ap={ap_id} change decision:"
                    f" cur_ch={cur_channel}, cur_ip={ip_dbm_current:.1f} dBm,"
                    f" target_ch={target}, target_ip={ip_dbm_target:.1f} dBm,"
                    f" improvement={improvement:.1f} dB"
                )
            if improvement < MIN_INTERFERENCE_IMPROVEMENT_DB:
                # Not enough improvement to justify churn
                continue

        # Either:
        #   - we triggered by counters only (ip_dbm_current is None), OR
        #   - we have a big enough dB improvement on target channel.
        proposals.append({
            "ap_id": ap_id,
            "type": "channel",
            "value": int(target),
            "reason": "fastloop_channel_change_due_to_interference",
        })
        _fastloop_last_change_ts[ap_id] = now
        _fastloop_last_target_channel[ap_id] = int(target)

        if cur_channel:
            channel_occupancy[cur_channel] = max(0, channel_occupancy[cur_channel] - 1)
        channel_occupancy[target] += 1

        if len(proposals) >= MAX_FASTLOOP_CHANGES_PER_STEP:
            break

    return proposals
