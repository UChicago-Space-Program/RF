#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AX.25 deframer component example
# Author: Daniel Estevez
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
from gnuradio import blocks, gr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import network
import satellites.components.deframers
import satellites.components.demodulators
import threading




class ax25_deframer(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "AX.25 deframer component example", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 48000
        self.center_frequency = center_frequency = 433350000

        ##################################################
        # Blocks
        ##################################################

        self.satellites_fsk_demodulator_0 = satellites.components.demodulators.fsk_demodulator(baudrate = 9600, samp_rate = samp_rate, iq = False, subaudio = False, options="")
        self.satellites_ax25_deframer_0 = satellites.components.deframers.ax25_deframer(g3ruh_scrambler=True, options="")
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.network_udp_sink_0 = network.udp_sink(gr.sizeof_char, 1, '127.0.0.1', 9000, 0, 1472, False)
        self.blocks_wavfile_source_0 = blocks.wavfile_source('/Users/kevinwu/Downloads/us01.wav', True)
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.satellites_ax25_deframer_0, 'out'), (self.blocks_message_debug_0, 'print_pdu'))
        self.msg_connect((self.satellites_ax25_deframer_0, 'out'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.blocks_wavfile_source_0, 0), (self.satellites_fsk_demodulator_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.satellites_fsk_demodulator_0, 0), (self.satellites_ax25_deframer_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_center_frequency(self):
        return self.center_frequency

    def set_center_frequency(self, center_frequency):
        self.center_frequency = center_frequency




def main(top_block_cls=ax25_deframer, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    tb.wait()


if __name__ == '__main__':
    main()
