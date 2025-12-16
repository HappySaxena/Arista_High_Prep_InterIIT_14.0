#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio
import Test_CNN_Predictor as CNN_Predictor  # embedded python module
import Test_epy_block_0 as epy_block_0  # embedded python block
import Test_epy_block_2 as epy_block_2  # embedded python block
import Test_epy_block_3 as epy_block_3  # embedded python block
import Test_epy_block_4 as epy_block_4  # embedded python block
import Test_epy_block_4_0ap3 as epy_block_4_0  # embedded python block
import Test_epy_block_4_0_0ap3 as epy_block_4_0_0 # embedded python block
import Test_epy_block_5 as epy_block_5  # embedded python block
import Test_epy_block_6 as epy_block_6  # embedded python block
import sip
import random


class Test(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "Test")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.wifi_channels = wifi_channels = {"1":2412000000,"2":2417000000,"3":2422000000,"4":2427000000,"5":2432000000,"6":2437000000,"7":2442000000,"8":2447000000,"9":2452000000,"10":2457000000,"11":2462000000,"12":2467000000,"13":2472000000,"21":5260000000,"22":5265000000,"23":5270000000,"24":5275000000,"25":5280000000,"26":5285000000,"27":5290000000,"28":5295000000,"36":5180000000,"40":5200000000,"44":5220000000,"48":5240000000}
        self.current_ch = current_ch = 3
        self.variable_0 = variable_0 = 0
        self.timestamps = timestamps = 0
        self.samp_rate = samp_rate = int(2000000)
        self.counter = counter = 1
        self.NON_DFS_5_GHz = NON_DFS_5_GHz = [36,40,44,48]
        self.NON_DFS_2_4_GHz = NON_DFS_2_4_GHz = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        #self.DFS_State = DFS_State = {21:"NOP", 22:"AVAILABLE", 23:"RADAR", 24:"AVAILABLE", 25:"AVAILABLE", 26:"NOP", 27:"AVAILABLE", 28:"AVAILABLE"}
        DFS_CHANNELS = [36, 40]
        dfs_state = ["AVAILABLE", "NOP", "RADAR"]
        dfs_prob = [0.70, 0.20, 0.10]
        self.DFS_State = DFS_State = {}
        
        for ch in DFS_CHANNELS:
            state = random.choices(dfs_state, weights=dfs_prob, k = 1)[0]
            self.DFS_State[ch] = state 

        self.Center_freq = Center_freq = wifi_channels[str(current_ch)]

        
        self.Center_freq = Center_freq = wifi_channels[str(current_ch)]

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_freq_sink_x_0_0_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            Center_freq, #fc
            samp_rate, #bw
            "Recieved", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0_0_0.set_update_time(0.1)
        self.qtgui_freq_sink_x_0_0_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0_0_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_0_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0_0_0.enable_grid(False)
        self.qtgui_freq_sink_x_0_0_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_0_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_0_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_0_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_0_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_0_0_win)
        self.iio_pluto_source_0 = iio.fmcomms2_source_fc32('ip:192.168.4.1' if 'ip:192.168.4.1' else iio.get_pluto_uri(), [True, True], 4096)
        self.iio_pluto_source_0.set_len_tag_key('packet_len')
        self.iio_pluto_source_0.set_frequency(Center_freq)
        self.iio_pluto_source_0.set_samplerate(samp_rate)
        self.iio_pluto_source_0.set_gain_mode(0, 'slow_attack')
        self.iio_pluto_source_0.set_gain(0, 50)
        self.iio_pluto_source_0.set_quadrature(True)
        self.iio_pluto_source_0.set_rfdc(True)
        self.iio_pluto_source_0.set_bbdc(True)
        self.iio_pluto_source_0.set_filter_params('Auto', '', 0, 0)
        self.epy_block_6 = epy_block_6.blk(short_win=256, noise_win=4096, alpha=0.3)
        self.epy_block_5 = epy_block_5.blk()
        self.epy_block_4_0_0 = epy_block_4_0_0.MAB_Controller(model_path='rbn_predictor_kalmanucb_optimized.pkl', hop_delay=0.005, parent=self)
        self.epy_block_4_0 = epy_block_4_0.blk(parent=self)
        self.epy_block_4 = epy_block_4.Interference_Predictor()
        self.epy_block_3 = epy_block_3.blk(alpha=0.1, base_thresh=3.0, cooldown=20, min_delta=1.0, slope_window=10)
        self.epy_block_2 = epy_block_2.blk()
        self.epy_block_0 = epy_block_0.blk(K=14, samp_rate=samp_rate)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 2048)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(2048)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.epy_block_2, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.epy_block_0, 0))
        self.connect((self.epy_block_0, 0), (self.epy_block_4_0_0, 0))
        self.connect((self.epy_block_0, 1), (self.epy_block_5, 0))
        self.connect((self.epy_block_2, 0), (self.epy_block_3, 0))
        self.connect((self.epy_block_3, 0), (self.epy_block_4_0, 0))
        self.connect((self.epy_block_3, 1), (self.epy_block_4_0, 1))
        self.connect((self.epy_block_4, 0), (self.epy_block_4_0, 3))
        self.connect((self.epy_block_4, 1), (self.epy_block_4_0, 7))
        self.connect((self.epy_block_4, 2), (self.epy_block_4_0, 8))
        self.connect((self.epy_block_4_0_0, 0), (self.epy_block_4_0, 2))
        self.connect((self.epy_block_4_0_0, 3), (self.epy_block_4_0, 6))
        self.connect((self.epy_block_4_0_0, 1), (self.epy_block_4_0, 4))
        self.connect((self.epy_block_4_0_0, 2), (self.epy_block_4_0, 5))
        self.connect((self.epy_block_5, 0), (self.epy_block_4, 0))
        self.connect((self.epy_block_6, 0), (self.epy_block_4_0, 9))
        self.connect((self.iio_pluto_source_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.epy_block_6, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.qtgui_freq_sink_x_0_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Test")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_wifi_channels(self):
        return self.wifi_channels

    def set_wifi_channels(self, wifi_channels):
        self.wifi_channels = wifi_channels
        self.set_Center_freq(self.wifi_channels[str(self.current_ch)])

    def get_current_ch(self):
        return self.current_ch

    def set_current_ch(self, current_ch):
        self.current_ch = current_ch
        self.set_Center_freq(self.wifi_channels[str(self.current_ch)])

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_timestamps(self):
        return self.timestamps

    def set_timestamps(self, timestamps):
        self.timestamps = timestamps

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.epy_block_0.samp_rate = self.samp_rate
        self.iio_pluto_source_0.set_samplerate(self.samp_rate)
        self.qtgui_freq_sink_x_0_0_0.set_frequency_range(self.Center_freq, self.samp_rate)

    def get_counter(self):
        return self.counter

    def set_counter(self, counter):
        self.counter = counter

    def get_NON_DFS_5_GHz(self):
        return self.NON_DFS_5_GHz

    def set_NON_DFS_5_GHz(self, NON_DFS_5_GHz):
        self.NON_DFS_5_GHz = NON_DFS_5_GHz

    def get_NON_DFS_2_4_GHz(self):
        return self.NON_DFS_2_4_GHz

    def set_NON_DFS_2_4_GHz(self, NON_DFS_2_4_GHz):
        self.NON_DFS_2_4_GHz = NON_DFS_2_4_GHz

    def get_DFS_State(self):
        return self.DFS_State

    def set_DFS_State(self, DFS_State):
        self.DFS_State = DFS_State

    def get_Center_freq(self):
        return self.Center_freq

    def set_Center_freq(self, Center_freq):
        self.Center_freq = Center_freq
        self.iio_pluto_source_0.set_frequency(self.Center_freq)
        self.qtgui_freq_sink_x_0_0_0.set_frequency_range(self.Center_freq, self.samp_rate)




def main(top_block_cls=Test, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

  #  timer = Qt.QTimer()
   # timer.start(1)
   # timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
