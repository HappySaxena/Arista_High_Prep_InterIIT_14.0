
#!/usr/bin/env python3
import time
import json
import paho.mqtt.client as mqtt
from params import params    # your params.py file (cleaned, no simulation)

BROKER = "192.168.50.1"      # RL controller                # Change if needed
TOPIC = f"rrm/summary/logs"

BUFFER_SIZE_K = 100  # Number of samples to consider per summary

# Paths for the 3 interference logs
LOG1 = "/home/happy/Downloads/interference_log1.json"
LOG2 = "/home/happy/Downloads/interference_log2.json"
LOG3 = "/home/happy/Downloads/interference_log3.json"

# Connect MQTT client
client = mqtt.Client(client_id=f"summary-sender")
client.connect(BROKER, 1883, 60)

def read_summary(path):
    """Safely run the params() function for a given log file."""
    try:
        summary = params(
            path=path,
            buffer_length=BUFFER_SIZE_K,
            Max_ap_range=10,
            mean_method="simple",
        )
        return summary
    except Exception as e:
        return {"error": str(e)}

def publish_summary(name, summary):
    """Publish summary to RL Controller with log name included."""
    payload = {
        "source": name,   # log1 / log2 / log3
        "summary": summary,
        "ts": time.time()
    }
    print(f"[PUBLISH] Sending {name} summary...")
    client.publish(TOPIC, json.dumps(payload))


print("=== SUMMARY SENDER STARTED ===")
print("RL Controller:", BROKER)
print("Sending summaries every 30 seconds...\n")

while True:
    # --- 1. SUMMARY FROM LOG1 ---
    summary1 = read_summary(LOG1)
    publish_summary("interference_log1", summary1)

    # --- 2. SUMMARY FROM LOG2 ---
    summary2 = read_summary(LOG2)
    publish_summary("interference_log2", summary2)

    # --- 3. SUMMARY FROM LOG3 ---
    summary3 = read_summary(LOG3)
    publish_summary("interference_log3", summary3)

    print("Cycle complete. Sleeping 30 seconds...\n")
    time.sleep(30)
