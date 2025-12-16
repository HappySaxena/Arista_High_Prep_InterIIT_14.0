#!/bin/bash

echo "======================================="
echo " AUTO VIRTUAL AP CREATOR (wlan0)"
echo "======================================="

echo "[+] Checking WiFi device..."

REAL_DEV=$(iw dev | awk '$1=="Interface"{print $2}' | head -n1)

if [ -z "$REAL_DEV" ]; then
    echo "❌ No WiFi interface found"
    exit 1
fi

echo "[✓] Found real device: $REAL_DEV"


echo "[+] Checking AP support..."
iw list | grep -q "* AP"
if [ $? -ne 0 ]; then
    echo "❌ Your WiFi card does NOT support AP mode"
    exit 1
fi
echo "[✓] AP mode supported"


echo "[+] Installing required packages..."
apt update -y
apt install -y hostapd dnsmasq


echo "[+] Creating virtual interface wlan0..."

ip link set $REAL_DEV down

iw dev $REAL_DEV interface add wlan0 type __ap

if ! ip link | grep -q wlan0; then
    echo "❌ Failed to create wlan0"
    exit 1
fi

echo "[✓] wlan0 created successfully"

ip link set wlan0 up
ip addr flush dev wlan0
ip addr add 192.168.4.1/24 dev wlan0

echo "[✓] wlan0 configured at 192.168.4.1"


echo "[+] Configuring hostapd..."

cat > /etc/hostapd/hostapd.conf <<EOF
interface=wlan0
driver=nl80211
ssid=MyLinuxAP
hw_mode=g
channel=6
ieee80211n=1
wmm_enabled=1
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=12345678
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

sed -i 's|#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd


echo "[+] Configuring dnsmasq..."

systemctl stop dnsmasq
mv /etc/dnsmasq.conf /etc/dnsmasq.conf.backup 2>/dev/null

cat > /etc/dnsmasq.conf <<EOF
interface=wlan0
dhcp-range=192.168.4.10,192.168.4.100,12h
EOF


echo "[+] Enabling IP Forwarding..."
sysctl -w net.ipv4.ip_forward=1


echo "[+] Stopping interfering services..."
systemctl stop systemd-resolved
systemctl disable systemd-resolved


echo "[+] Starting AP services..."

systemctl unmask hostapd
systemctl enable hostapd
systemctl restart hostapd
systemctl restart dnsmasq


sleep 2

echo "======================================="
echo " ✅ ACCESS POINT CREATED SUCCESSFULLY "
echo "======================================="
echo " WiFi Name : MyLinuxAP"
echo " Password  : 12345678"
echo " Gateway   : 192.168.4.1"
echo "======================================="

iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE 2>/dev/null

echo "[✓] Done. Try connecting your phone now."


