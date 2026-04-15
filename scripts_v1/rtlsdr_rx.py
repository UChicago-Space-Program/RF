#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# RTL-SDR RX flowgraph for the BER/FER test harness.
#
# Differences from the original Plutosdr_rx_sample.py:
#   - The Pluto source (iio.fmcomms2_source_fc32) is replaced with an
#     osmosdr.source for the RTL-SDR. RTL-SDR is RX-only, which is why
#     we keep the Pluto on the TX side and use RTL-SDR here.
#   - RTL-SDR doesn't support 576 kHz natively (its valid rates are
#     ~225-300 kHz and ~900 kHz - 3.2 MHz), so we run the RTL-SDR at
#     1.152 MHz (= 2 * 576 kHz) and decimate by 2 inside the existing
#     freq_xlating_fir_filter. The rest of the chain still operates
#     at 576 kHz, unchanged.
#   - All Pluto-specific knobs (set_quadrature, set_rfdc, set_bbdc,
#     set_filter_params, set_len_tag_key, set_bandwidth) are gone.
#     RTL-SDR has a different gain model: tuner gain + IF gain +
#     baseband gain, plus PPM frequency correction.
#   - socket_pdu changed from TCP_SERVER:52001 to UDP_CLIENT pointing
#     at 127.0.0.1:52002 so the BER script can recv each AX.25 PDU
#     as one UDP datagram. The port is different from the TX side
#     (52001) because both flowgraphs run on the same machine.
#
# The Qt GUI freq/time sinks are kept — they're useful for confirming
# the link is alive before running a BER trial.
#
# Bug fix (2026-04-14):
#   - pfb_clock_sync_fff (RRC matched filter) replaced with
#     digital.clock_recovery_mm_ff (Mueller-Müller TED). The pfb TED
#     assumes an RRC pulse shape; GFSK uses a Gaussian pulse. The
#     mismatch caused poor timing convergence (~25% decode rate).
#     MM TED is signal-shape agnostic and achieves ~99% decode rate.
#   - Verified working settings: --gain 40, --rx-offset 0.
#     (RTL-SDR Blog V4 TCXO is accurate to <1 PPM at 433 MHz; no
#      manual frequency correction needed.)

import math
import signal
import sys
import threading
from argparse import ArgumentParser

import osmosdr
from gnuradio import analog, blocks, filter, gr, network, pdu, qtgui
from gnuradio import digital
from gnuradio.eng_arg import eng_float
from gnuradio.fft import window
from gnuradio.filter import firdes
from PyQt5 import Qt, sip
import satellites.components.deframers


class rtlsdr_rx(gr.top_block, Qt.QWidget):
    def __init__(self,
                 samp_rate_in=1_152_000,   # RTL-SDR rate (must be in 0.9-3.2 MHz)
                 decim=2,                  # decimate to 576 kHz for the demod chain
                 center_frequency=433_350_000,
                 fsk_deviation_hz=3000,
                 rx_offset_hz=0,
                 rtl_ppm=0,
                 rtl_gain=40,
                 rtl_if_gain=20,
                 rtl_bb_gain=20,
                 udp_dest_host="127.0.0.1",
                 udp_dest_port=52002):
        gr.top_block.__init__(self, "RTL-SDR RX (BER harness)",
                              catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("RTL-SDR RX (BER harness)")

        # Minimal Qt scaffolding so freq/time sinks have somewhere to live.
        self.top_layout = Qt.QVBoxLayout(self)
        self.flowgraph_started = threading.Event()

        # ---------------- Variables ----------------
        self.samp_rate_in = samp_rate_in
        self.decim = decim
        self.samp_rate = samp_rate_in // decim   # post-decim rate (576000)
        self.sps = 60
        self.center_frequency = center_frequency
        self.fsk_deviation_hz = fsk_deviation_hz
        self.rx_offset_hz = rx_offset_hz

        # ---------------- RTL-SDR source -----------
        # args="numchan=1 rtl=0" picks the first RTL-SDR. If you have
        # multiple, change rtl=0 to rtl=1, or pass a serial like rtl=00000001.
        self.osmosdr_source_0 = osmosdr.source(args="numchan=1 rtl=0")
        self.osmosdr_source_0.set_sample_rate(samp_rate_in)
        self.osmosdr_source_0.set_center_freq(center_frequency, 0)
        self.osmosdr_source_0.set_freq_corr(rtl_ppm, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(False, 0)   # manual gain
        self.osmosdr_source_0.set_gain(rtl_gain, 0)
        self.osmosdr_source_0.set_if_gain(rtl_if_gain, 0)
        self.osmosdr_source_0.set_bb_gain(rtl_bb_gain, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(0, 0)

        # ---------------- DSP chain ----------------
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(
            -100, 1e-4, 0, True)

        # The freq_xlating filter does the heavy lifting now: it
        # frequency-translates AND decimates samp_rate_in -> samp_rate.
        # Taps are designed at the input rate.
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccf(
            decim,
            firdes.low_pass(1.0, samp_rate_in, 25_000, 8_000),
            rx_offset_hz,
            samp_rate_in)

        self.analog_quadrature_demod_cf_0 = analog.quadrature_demod_cf(
            self.samp_rate / (2 * math.pi * fsk_deviation_hz))

        self.low_pass_filter_0 = filter.fir_filter_fff(
            1,
            firdes.low_pass(1, self.samp_rate, 15_000, 5_000,
                            window.WIN_HAMMING, 6.76))

        # Mueller-Müller TED clock recovery works better than pfb_clock_sync
        # for GFSK because it does not assume a specific (RRC) pulse shape.
        # omega = initial samples per symbol; gain_omega and gain_mu control
        # how fast the loop tracks sample rate and timing phase respectively.
        # omega_relative_limit caps the allowed sample-rate deviation to ±0.5%.
        self.digital_clock_recovery_mm_0 = digital.clock_recovery_mm_ff(
            self.sps * 1.0,          # omega: initial samples per symbol (60)
            0.25 * 0.175 * 0.175,    # gain_omega ≈ 0.25 × gain_mu²
            0.5,                     # mu: initial timing offset (0–1)
            0.175,                   # gain_mu: timing tracking loop gain
            0.005)                   # omega_relative_limit: ±0.5% rate deviation

        # ---------------- AX.25 deframer + egress --
        self.satellites_ax25_deframer_0 = \
            satellites.components.deframers.ax25_deframer(
                g3ruh_scrambler=False, options="")

        # UDP_CLIENT: every AX.25 PDU is sent as one UDP datagram to
        # the BER test script's listener.
        self.network_socket_pdu_0 = network.socket_pdu(
            'UDP_CLIENT', udp_dest_host, str(udp_dest_port), 10000, False)

        self.blocks_message_debug_1 = blocks.message_debug(
            True, gr.log_levels.info)  # was debug — debug is suppressed by default GR log level

        # ---------------- Qt GUI sinks (monitoring) -
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            4096, window.WIN_BLACKMAN_hARRIS,
            0, samp_rate_in, "RTL-SDR input", 1, None)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.top_layout.addWidget(
            sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget))

        self.qtgui_freq_sink_x_1 = qtgui.freq_sink_c(
            4096, window.WIN_BLACKMAN_hARRIS,
            0, self.samp_rate, "After xlate/decim", 1, None)
        self.qtgui_freq_sink_x_1.set_y_axis(-140, 10)
        self.top_layout.addWidget(
            sip.wrapinstance(self.qtgui_freq_sink_x_1.qwidget(), Qt.QWidget))

        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024, self.samp_rate, "Symbol sync output", 1, None)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)
        self.top_layout.addWidget(
            sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget))

        # ---------------- Connections --------------
        self.connect((self.osmosdr_source_0, 0),
                     (self.analog_pwr_squelch_xx_0, 0))
        self.connect((self.analog_pwr_squelch_xx_0, 0),
                     (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.analog_pwr_squelch_xx_0, 0),
                     (self.freq_xlating_fir_filter_xxx_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0),
                     (self.qtgui_freq_sink_x_1, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0),
                     (self.analog_quadrature_demod_cf_0, 0))
        self.connect((self.analog_quadrature_demod_cf_0, 0),
                     (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0),
                     (self.digital_clock_recovery_mm_0, 0))
        self.connect((self.digital_clock_recovery_mm_0, 0),
                     (self.qtgui_time_sink_x_0, 0))
        self.connect((self.digital_clock_recovery_mm_0, 0),
                     (self.satellites_ax25_deframer_0, 0))

        self.msg_connect((self.satellites_ax25_deframer_0, 'out'),
                         (self.network_socket_pdu_0, 'pdus'))
        self.msg_connect((self.satellites_ax25_deframer_0, 'out'),
                         (self.blocks_message_debug_1, 'print_pdu'))

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()


def main():
    parser = ArgumentParser()
    parser.add_argument("--samp-rate-in", type=eng_float, default=1_152_000,
                        help="RTL-SDR sample rate (Hz). Must be 0.9-3.2 MHz.")
    parser.add_argument("--decim", type=int, default=2,
                        help="Decimation factor inside freq_xlating filter.")
    parser.add_argument("--freq", type=eng_float, default=433_350_000)
    parser.add_argument("--rx-offset", type=eng_float, default=0)
    parser.add_argument("--ppm", type=int, default=0,
                        help="RTL-SDR frequency correction in ppm.")
    parser.add_argument("--gain", type=float, default=40)
    parser.add_argument("--if-gain", type=float, default=20)
    parser.add_argument("--bb-gain", type=float, default=20)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=52002)
    args = parser.parse_args()

    qapp = Qt.QApplication(sys.argv)
    tb = rtlsdr_rx(
        samp_rate_in=int(args.samp_rate_in),
        decim=args.decim,
        center_frequency=int(args.freq),
        rx_offset_hz=int(args.rx_offset),
        rtl_ppm=args.ppm,
        rtl_gain=args.gain,
        rtl_if_gain=args.if_gain,
        rtl_bb_gain=args.bb_gain,
        udp_dest_host=args.udp_host,
        udp_dest_port=args.udp_port,
    )
    tb.start()
    tb.flowgraph_started.set()
    tb.show()

    def sig_handler(*_):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    qapp.exec_()


if __name__ == "__main__":
    main()
