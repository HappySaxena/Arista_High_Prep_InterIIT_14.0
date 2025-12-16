#!/usr/bin/env python3
import os
import time
import statistics
from scapy.all import sniff, TCP, IP

# ---------------- CONFIG ----------------
IFACE = "wlan0"          # interface to sniff
IPERF_PORT = 5201        # destination port for iperf3
MAX_SAMPLES = 50         # how many RTT values to collect before stopping

# ----------------------------------------
# (src, dst, tsval) -> send timestamp (seconds)
sent_ts = {}

# Store RTT samples in milliseconds
rtts = []

# Stop flag
done = False

# Counters for loss estimation
total_ts_outgoing = 0    # client->server packets with TCP TS
total_ts_acked = 0       # those that got echoed back (RTT computed)
# ----------------------------------------


def print_final_stats():
    """Print final RTT and loss statistics once."""
    global total_ts_outgoing, total_ts_acked

    if len(rtts) == 0:
        print("No RTT samples collected.")
        return

    # Median RTT
    rtt_median = statistics.median(rtts)

    # P95 RTT (if we have at least 2 samples)
    if len(rtts) >= 2:
        rtt_p95 = statistics.quantiles(rtts, n=100)[94]
    else:
        rtt_p95 = rtts[0]

    # ----- Loss estimation -----
    losses = max(total_ts_outgoing - total_ts_acked, 0)
    if total_ts_outgoing > 0:
        loss_rate = losses / total_ts_outgoing          # 0..1
        loss_variance = loss_rate * (1.0 - loss_rate)   # Bernoulli variance
    else:
        loss_rate = 0.0
        loss_variance = 0.0

    print("\n===== FINAL RTT RESULTS =====")
    print(f"Samples       : {len(rtts)}")
    print(f"RTT Median    : {rtt_median:.2f} ms")
    print(f"RTT P95       : {rtt_p95:.2f} ms")
    print(f"Loss rate     : {loss_rate * 100:.2f} %")
    print(f"Loss variance : {loss_variance:.6f}")
    print("=================================\n")


def parse_rtt(pkt):
    """Process each sniffed packet and compute RTT when possible."""
    global done, total_ts_outgoing, total_ts_acked

    if done:
        return

    # We only care about TCP + IP packets
    if not pkt.haslayer(TCP) or not pkt.haslayer(IP):
        return

    ip = pkt[IP]
    tcp = pkt[TCP]
    now = time.time()

    src = ip.src
    dst = ip.dst
    sport = tcp.sport
    dport = tcp.dport

    # Extract TCP timestamps (TSval, TSecr) if present
    tsval = None
    tsecr = None
    for opt in tcp.options:
        if opt[0] == "Timestamp":
            tsval, tsecr = opt[1]
            break

    # ---------- 1) CLIENT → SERVER direction ----------
    # Client sends data to iperf server on port IPERF_PORT
    if dport == IPERF_PORT:
        if tsval is not None:
            key = (src, dst, tsval)
            # Only record once per (src, dst, tsval)
            if key not in sent_ts:
                sent_ts[key] = now
                total_ts_outgoing += 1
        return

    # ---------- 2) SERVER → CLIENT direction ----------
    # Server replies from IPERF_PORT and echoes the timestamp in TSecr
    if sport == IPERF_PORT and tsecr is not None:
        key = (dst, src, tsecr)

        start = sent_ts.pop(key, None)
        if start is None:
            # We never saw the original TSval or it expired
            return

        rtt_ms = (now - start) * 1000.0
        rtts.append(rtt_ms)
        total_ts_acked += 1

        # Stop once we have enough samples
        if len(rtts) >= MAX_SAMPLES:
            done = True
            print_final_stats()
            os._exit(0)      # hard exit to stop sniff immediately


if __name__ == "__main__":
    print(f"Starting passive RTT monitor (one-shot) on {IFACE} ...")
    sniff(iface=IFACE, prn=parse_rtt, store=0)
