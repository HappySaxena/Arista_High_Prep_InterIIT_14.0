#!/bin/bash

echo "==============================="
echo " CONFIGURING WLAN0 AS AP"
echo "==============================="

AP_IF="wlan0"
NET_IF="wlp61s0"

# Stop conflicting services
systemctl stop hostapd dnsmasq NetworkManager
systemctl disable systemd-resolved
systemctl stop systemd-resolved

# Configure wlan0
ip link set $AP_IF down
ip addr flush dev $AP_IF
ip addr add 192.168.4.1/24 dev $AP_IF
ip link set $AP_IF up

iw dev $AP_IF set txpower fixed 1300

echo "[✓] wlan0 configured with IP 192.168.4.1"

# Create hostapd config
cat > /etc/hostapd/hostapd.conf <<EOF
ctrl_interface=/var/run/hostapd
ctrl_interface_group=0
interface=wlan0
driver=nl80211
ssid=Happy_AP
hw_mode=g
channel=7
ieee80211n=1
ht_capab=[HT40+]
country_code=US
rts_threshold=500
wmm_enabled=1
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=12345678
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

sed -i 's|#DAEMON_CONF=""|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd

# Configure dnsmasq
mv /etc/dnsmasq.conf /etc/dnsmasq.conf.backup 2>/dev/null

cat > /etc/dnsmasq.conf <<EOF
interface=wlan0
dhcp-range=192.168.4.10,192.168.4.100,12h
EOF

# Enable IP Forwarding
sysctl -w net.ipv4.ip_forward=1

# Setup NAT internet sharing
iptables -t nat -F
iptables -F
iptables -t nat -A POSTROUTING -o $NET_IF -j MASQUERADE
iptables -A FORWARD -i $NET_IF -o $AP_IF -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i $AP_IF -o $NET_IF -j ACCEPT

# Start services
systemctl unmask hostapd
systemctl enable hostapd
systemctl restart hostapd
systemctl restart dnsmasq

echo "==============================="
echo " ✅ HOTSPOT ACTIVE"
echo "==============================="
echo " WIFI NAME  : Happy_AP"
echo " PASSWORD   : 12345678"
echo " GATEWAY    : 192.168.4.1"
echo "==============================="
echo " Connect your phone now"
