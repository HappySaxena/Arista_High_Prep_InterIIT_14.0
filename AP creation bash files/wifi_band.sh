#!/bin/bash

AP_SCRIPT="start_ap.sh"

if [ ! -f "$AP_SCRIPT" ]; then
    echo "Error: $AP_SCRIPT not found!"
    exit 1
fi

if [ "$1" == "2.4" ]; then
    echo "Switching start_ap.sh to 2.4 GHz..."

    sed -i 's/hw_mode=.*/hw_mode=g/' $AP_SCRIPT
    sed -i 's/channel=.*/channel=3/' $AP_SCRIPT
    sed -i 's/ht_capab=.*/ht_capab=\[HT40+\]/' $AP_SCRIPT

    # remove 5 GHz VHT lines if present
    sed -i '/ieee80211ac=/d' $AP_SCRIPT
    sed -i '/vht_oper_chwidth=/d' $AP_SCRIPT
    sed -i '/vht_oper_centr_freq_seg0_idx=/d' $AP_SCRIPT

elif [ "$1" == "5" ]; then
    echo "Switching start_ap.sh to 5 GHz..."

    sed -i 's/hw_mode=.*/hw_mode=a/' $AP_SCRIPT
    sed -i 's/channel=.*/channel=36/' $AP_SCRIPT

    # upgrade to 802.11ac/VHT automatically
    grep -q "ieee80211ac=" $AP_SCRIPT || echo 'ieee80211ac=1' >> $AP_SCRIPT
    grep -q "vht_oper_chwidth=" $AP_SCRIPT || echo 'vht_oper_chwidth=1' >> $AP_SCRIPT
    grep -q "vht_oper_centr_freq_seg0_idx=" $AP_SCRIPT || echo 'vht_oper_centr_freq_seg0_idx=42' >> $AP_SCRIPT

    # ensure 802.11n stays enabled
    sed -i 's/ieee80211n=.*/ieee80211n=1/' $AP_SCRIPT

else
    echo "Usage: $0 {2.4|5}"
    exit 1
fi

echo "Done."

