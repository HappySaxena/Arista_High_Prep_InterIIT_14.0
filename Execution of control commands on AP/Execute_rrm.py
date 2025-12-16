#!/usr/bin/env python3
import subprocess, json, time, paho.mqtt.client as mqtt
import uuid, re
# from params import params
# INTERFERENCE_LOG_PATH=
# BUFFER_SIZE_K=25
BROKER = "192.168.50.1"       # Controller laptop IP
AP_ID  = "ap1"                # Change per AP

START_AP  = "/home/happy/start_ap.sh"    # LOCAL script
WIFI_BAND = "/home/happy/wifi_band.sh"   # LOCAL script to switch 2.4GHz / 5GHz

client = mqtt.Client(client_id=f"{AP_ID}-{uuid.uuid4()}")
client.connect(BROKER, 1883, 60)

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
def run(cmd):
    """Run shell command with print + error handling."""
    print(f"[EXEC] {cmd}")
    try:
        return subprocess.check_output(cmd, shell=True).decode()
    except Exception as e:
        print("[ERROR]", e)
        return None


def safe_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode()
    except:
        return "ERROR"


def get_current_band():
    """Return '2.4' or '5' based on channel."""
    try:
        iw = subprocess.check_output("iw dev wlan0 info", shell=True).decode()
    except:
        return "unknown"

    m = re.search(r"channel\s+(\d+)", iw)
    if not m:
        return "unknown"

    ch = int(m.group(1))

    if 1 <= ch <= 13:
        return "2.4"
    else:
        return "5"


# ==========================================================
# OBSS-PD → TX POWER MAPPING (MATHEMATICAL MODEL)
# ==========================================================
def obss_to_txpower(obss_pd):
    """
    Smooth linear mapping:
    OBSS-PD [-82 .. -62]  →  TX [10 .. 20] dBm
    """
    OBSS_MIN = -82.0
    OBSS_MAX = -62.0
    TX_MIN   = 10.0
    TX_MAX   = 20.0

    obss_pd = max(OBSS_MIN, min(OBSS_MAX, obss_pd))
    norm = (obss_pd - OBSS_MIN) / (OBSS_MAX - OBSS_MIN)
    tx = TX_MIN + norm * (TX_MAX - TX_MIN)

    return round(tx)


# ==========================================================
# EXECUTE ACTIONS RECEIVED FROM CONTROLLER
# ==========================================================
def apply_action(action, value):
    print(f"\n[ACTION RECEIVED] {action} = {value}")

    current_band = get_current_band()

    # ------------------------------------------------------
    # 1. CHANGE CHANNEL
    # ------------------------------------------------------
    if action == "set_channel":
        new_ch = int(value)
        new_band = "2.4" if 1 <= new_ch <= 13 else "5"

        # Band switch if needed
        if new_band != current_band:
            run(f"sudo {WIFI_BAND} {new_band}")
            time.sleep(1)
            run(f"sudo {START_AP}")
            time.sleep(5)

        # Modify channel in start_ap.sh
        run(f"sudo sed -i 's/^channel=.*/channel={new_ch}/' {START_AP}")

        # Restart AP
        run(f"sudo {START_AP}")
        return

    # ------------------------------------------------------
    # 2. CHANGE BANDWIDTH
    # ------------------------------------------------------
    if action == "set_bw":
        bw = int(value)

        if bw == 20:
            ht = "[HT20]"
        elif bw == 40:
            ht = "[HT40+]"
        elif bw == 80:
            ht = "[HT80+]"
        else:
            print("[ERROR] Unsupported BW:", bw)
            return

        run(f"sudo sed -i 's/^ht_capab=.*/ht_capab={ht}/' {START_AP}")
        run(f"sudo {START_AP}")
        return

    # ------------------------------------------------------
    # 3. CHANGE TX POWER (dBm)
    # ------------------------------------------------------
    if action == "set_tx_power":
        tx = int(value)
        tx_fixed = tx * 100  # hostapd format

        run(f"sudo sed -i 's/txpower fixed [0-9]\\+/txpower fixed {tx_fixed}/' {START_AP}")
        run(f"sudo {START_AP}")
        return

    # ------------------------------------------------------
    # 4. OBSS-PD EMULATION → MAP TO TX POWER
    # ------------------------------------------------------
    if action == "set_obss_pd":
        obss = float(value)
        tx = obss_to_txpower(obss)
        tx_fixed = tx * 100

        print(f"[OBSS-MAP] OBSS={obss} → TX={tx} dBm")

        run(f"sudo sed -i 's/txpower fixed [0-9]\\+/txpower fixed {tx_fixed}/' {START_AP}")
        run(f"sudo {START_AP}")
        return


# ==========================================================
# MQTT CALLBACKS
# ==========================================================
def on_message(c, u, msg):
    try:
        data = json.loads(msg.payload.decode())
        print("\n[CMD RECEIVED RAW]:", data)

        action = data.get("action")
        value  = data.get("value")

        if action:
            apply_action(action, value)

    except Exception as e:
        print("[ERROR parsing MQTT message]", e)


client.subscribe(f"rrm/actions/{AP_ID}")
client.on_message = on_message
client.loop_start()

# def run_rtt_safe(timeout_sec=5):
#     """Run passive_rtt.py safely with timeout. Never crash AP script."""
#     try:
#         # Use Popen so timeout does NOT kill the main process
#         proc = subprocess.Popen(
#             ["sudo", "python3", "passive_rtt.py"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )

#         try:
#             out, err = proc.communicate(timeout=timeout_sec)
#             if out.strip():
#                 return out
#             if err.strip():
#                 return f"RTT_ERROR: {err}"
#             return "NO_RTT_OUTPUT"

#         except subprocess.TimeoutExpired:
#             proc.kill()
#             return "RTT_TIMEOUT"

#     except Exception as e:
#         return f"RTT_FATAL_ERROR: {e}"


# ==========================================================
# TELEMETRY
# ==========================================================
def get_telemetry_raw():
    tel={
        "iw_info": safe_cmd("iw dev wlan0 info"),
        "station_dump": safe_cmd("iw dev wlan0 station dump"),
    #     "rtt": run_rtt_safe(timeout_sec=5)}
    # try:
    #     tel["channel_summary"]=params.params(
    #         path=INTERFERENCE_LOG_PATH,
    #         buffer_length=BUFFER_SIZE_K,
    #         Max_ap_range=10
    #         mean_method="simple",
    }
    # except Exception as e:
    #     tel["channel_summary_error"]=str(e)
    return tel        

    


# ==========================================================
# MAIN TELEMETRY LOOP
# ==========================================================
while True:
    packet = {
        "ap_id": AP_ID,
        "telemetry": get_telemetry_raw(),
        "ts": time.time()
    }

    client.publish(f"rrm/telemetry/{AP_ID}", json.dumps(packet))
    print("Published telemetry packet")
    time.sleep(30)
