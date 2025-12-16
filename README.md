# Arista_High_Prep_InterIIT_14.0  
### Intelligent Radio Resource Management (RRM)  
**Inter IIT Tech Meet 14.0 – Arista Networks (Problem Statement H4)**

---

## 📌 Overview
This repository contains **Our complete solution** for the *Intelligent Radio Resource Management (RRM)* problem statement proposed by **Arista Networks** at **Inter IIT Tech Meet 14.0**.

The project focuses on building a **safe, explainable, and production-ready Wi-Fi RRM system** using:
- SDR-based continuous spectrum sensing  
- Multi-timescale control loops  
- Conservative Reinforcement Learning (CQL) with safety Guardrails 
- Graph-based interference modeling  
- Client-centric QoE inference  

The solution is validated on a **real Linux-based Virtual AP testbed**, not just simulation.

---

## 🧠 System Architecture
The system is designed around **three coordinated control loops**, each operating at a different timescale.

### 1️⃣ Fast Loop (Seconds – Minutes)
- Acts as a **real-time safety layer**
- Reacts to:
  - DFS radar hits
  - Sudden interference spikes
  - Channel congestion
- Automatically steers APs away from unsafe spectrum
- No manual intervention required

---

### 2️⃣ Slow Loop (Hours – Days)
- Performs **global RF optimization**
- Uses:
  - Interference graph construction
  - DSATUR graph coloring
  - GNN-based Q-learning with **Conservative Q-Learning (CQL)**
- Optimizes:
  - Channel
  - Transmit power
  - Bandwidth
  - OBSS-PD thresholds
- Learns **offline from logs** under strict safety guardrails

---

### 3️⃣ Event Loop
- Handles **context-aware scenarios** such as:
  - Examination halls
  - Meeting rooms
  - High-interference environments
- Ensures network stability during sudden environmental or policy changes

---

## 📡 SDR-Based Sensing Orchestrator
To overcome limitations of AP-side scanning, we implemented a **dedicated sensing radio per AP** using **PlutoSDR and GNU Radio**.

### Key Capabilities
- Continuous spectrum monitoring (2.4 GHz & 5 GHz)
- Detection of non-Wi-Fi interference:
  - BLE
  - Zigbee
  - Microwave
  - Continuous Wave (CW)
- Noise-floor change detection using **CUSUM / EWMA**
- Multi-armed bandit (Kalman-UCB) based scan scheduling
- Structured JSON telemetry output for the RRM controller

---

## 👥 Advanced Client View (Without 802.11mc)
Since deployment hardware lacked IEEE 802.11mc (FTM), a **passive client-view framework** was designed using TCP timestamps.

For each client, the system derives:
- Median RTT
- P95 RTT
- Loss rate and loss variance
- RSSI-based spatial bins:
  - **Near** (> −45 dBm)
  - **Mid** (−65 to −45 dBm)
  - **Edge** (−75 to −65 dBm)

This enables **QoE-aware decision making without PHY-layer support**.

---

## 🛡️ Safe Reinforcement Learning (CQL)
The slow loop uses **Conservative Q-Learning (CQL)** to ensure:
- No unsafe online exploration
- Pessimistic Q-values for unseen actions
- Strict KPI guardrails

### Reward Balances
- Throughput and coverage
- Fairness across APs
- Retry minimization
- Configuration churn control

---

## 📁 Repository Structure
```text
├── AP creation bash files/
│   └── Scripts for Linux-based Virtual AP creation
│
├── Advanced Client View/
│   └── Passive RTT inference and client QoE estimation
│
├── Detailed design document with APIs/
│   └── Architecture diagrams and API definitions
│
├── Execution of control commands on AP/
│   └── Channel, power, bandwidth, OBSS-PD control logic
│
├── Multi Timescale control loops/
│   ├── Fast loop
│   ├── Slow loop (RL-based)
│   └── Event loop
│
├── SensingOrchestrator/
│   └── SDR and GNU Radio sensing pipelines
│
├── Presentation ppt.pdf
│   └── Final Inter IIT presentation
│
├── Report_Team_24.pdf
│   └── Complete end-term technical report
│
├── plot_acceptance_rate.png
├── plot_inference_distribution.png
│
└── README.md
```
## 🧪 Experimental Setup
- **APs**: Linux laptops (Wi-Fi 6 capable)
- **SDRs**: Akademika Pluto-SDR
- **Interference Sources**: BLE, Zigbee, Microwave, Continuous Wave (CW)
- **Clients**: Windows, Android, IoT, and Edge devices
- **Environment**: Real RF conditions evaluated over multiple days

---

## 📊 Key Outcomes
- Real-time environments exhibit **higher uncertainty and variability** compared to simulation
- **SDR-based sensing** significantly improves interference awareness and detection accuracy
- **Conservative Q-Learning (CQL)** enables safe and stable optimization using offline data
- **Event-aware control** prevents QoE degradation in critical and high-priority scenarios
