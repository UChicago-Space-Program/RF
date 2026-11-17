#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OPS-SAT TX (WIP)
# Author: Kevin Wu
# Description: OPS-SAT TX chain divided into GNU Radio DSP blocks. Imports ops_sat_framing.py for CSP + OPS-SAT framing bits.
# GNU Radio version: 3.10.12.0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import ops_sat_tx_wip_ops_stage as ops_stage  # embedded python module
import threading




class ops_sat_tx_wip(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "OPS-SAT TX (WIP)", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 6
        self.output_rate = output_rate = 250000
        self.offset_hz = offset_hz = 40000
        self.mod_index = mod_index = 1.0416666667
        self.invert_symbols = invert_symbols = False
        self.baud = baud = 9600
        self.symbol_table = symbol_table = ([1.0, -1.0] if invert_symbols else [-1.0, 1.0])
        self.sensitivity = sensitivity = 3.141592653589793*mod_index/sps
        self.samp_rate = samp_rate = 250e3
        self.repeat = repeat = 20
        self.preamble_flags = preamble_flags = 50
        self.postamble_flags = postamble_flags = 5
        self.phase_inc = phase_inc = 6.283185307179586*offset_hz/output_rate
        self.payload_text = payload_text = None
        self.payload_mode = payload_mode = "csp"
        self.payload_len = payload_len = 62
        self.payload_hex = payload_hex = ""
        self.output_path = output_path = "/Users/kevinwu/Downloads/generated_osat_csp_from_blockified_grc.cf32"
        self.nrzi_initial = nrzi_initial = 0
        self.interp_out = interp_out = 625
        self.gap_ms = gap_ms = 25.0
        self.freq = freq = 437.160e6
        self.decim_out = decim_out = 144
        self.csp_src = csp_src = 5
        self.csp_sport = csp_sport = 0
        self.csp_priority = csp_priority = 3
        self.csp_flags = csp_flags = 0x00
        self.csp_dst = csp_dst = 10
        self.csp_dport = csp_dport = 31
        self.csp_data_text = csp_data_text = None
        self.csp_data_hex = csp_data_hex = ""
        self.csp_crc_include_header = csp_crc_include_header = False
        self.csp_crc32c = csp_crc32c = False
        self.bt = bt = 0.5
        self.baseband_rate = baseband_rate = baud*sps
        self.amplitude = amplitude = 0.25

        ##################################################
        # Blocks
        ##################################################

        self.soapy_plutosdr_sink_0 = None
        dev = 'driver=plutosdr'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_plutosdr_sink_0 = soapy.sink(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_plutosdr_sink_0.set_sample_rate(0, samp_rate)
        self.soapy_plutosdr_sink_0.set_bandwidth(0, 0.0)
        self.soapy_plutosdr_sink_0.set_frequency(0, freq)
        self.soapy_plutosdr_sink_0.set_gain(0, min(max(20, 0.0), 89.0))
        self.rational_resampler_ccc_0 = filter.rational_resampler_ccc(
                interpolation=interp_out,
                decimation=decim_out,
                taps=[],
                fractional_bw=0)
        self.interp_fir_filter_fff_0 = filter.interp_fir_filter_fff(sps, ops_stage.gmsk_taps(bt, sps))
        self.interp_fir_filter_fff_0.declare_sample_delay(0)
        self.digital_chunks_to_symbols_bf_0 = digital.chunks_to_symbols_bf(symbol_table, 1)
        self.blocks_vector_source_gate = blocks.vector_source_c(ops_stage.make_gate(repeat, payload_len, payload_mode, payload_text, payload_hex, preamble_flags, postamble_flags, gap_ms, nrzi_initial, sps, csp_data_text, csp_data_hex, csp_priority, csp_src, csp_dst, csp_dport, csp_sport, csp_flags, csp_crc32c, csp_crc_include_header), True, 1, [])
        self.blocks_vector_source_bits = blocks.vector_source_b(ops_stage.make_bitstream(repeat, payload_len, payload_mode, payload_text, payload_hex, preamble_flags, postamble_flags, gap_ms, nrzi_initial, sps, csp_data_text, csp_data_hex, csp_priority, csp_src, csp_dst, csp_dport, csp_sport, csp_flags, csp_crc32c, csp_crc_include_header), True, 1, [])
        self.blocks_rotator_cc_0 = blocks.rotator_cc(phase_inc, False)
        self.blocks_multiply_gate = blocks.multiply_vcc(1)
        self.blocks_multiply_const_amp = blocks.multiply_const_cc(amplitude)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, output_path, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(sensitivity)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_gate, 0))
        self.connect((self.blocks_multiply_const_amp, 0), (self.rational_resampler_ccc_0, 0))
        self.connect((self.blocks_multiply_gate, 0), (self.blocks_multiply_const_amp, 0))
        self.connect((self.blocks_rotator_cc_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_rotator_cc_0, 0), (self.soapy_plutosdr_sink_0, 0))
        self.connect((self.blocks_vector_source_bits, 0), (self.digital_chunks_to_symbols_bf_0, 0))
        self.connect((self.blocks_vector_source_gate, 0), (self.blocks_multiply_gate, 1))
        self.connect((self.digital_chunks_to_symbols_bf_0, 0), (self.interp_fir_filter_fff_0, 0))
        self.connect((self.interp_fir_filter_fff_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.rational_resampler_ccc_0, 0), (self.blocks_rotator_cc_0, 0))


    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_baseband_rate(self.baud*self.sps)
        self.set_sensitivity(3.141592653589793*self.mod_index/self.sps)
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.interp_fir_filter_fff_0.set_taps(ops_stage.gmsk_taps(self.bt, self.sps))

    def get_output_rate(self):
        return self.output_rate

    def set_output_rate(self, output_rate):
        self.output_rate = output_rate
        self.set_phase_inc(6.283185307179586*self.offset_hz/self.output_rate)

    def get_offset_hz(self):
        return self.offset_hz

    def set_offset_hz(self, offset_hz):
        self.offset_hz = offset_hz
        self.set_phase_inc(6.283185307179586*self.offset_hz/self.output_rate)

    def get_mod_index(self):
        return self.mod_index

    def set_mod_index(self, mod_index):
        self.mod_index = mod_index
        self.set_sensitivity(3.141592653589793*self.mod_index/self.sps)

    def get_invert_symbols(self):
        return self.invert_symbols

    def set_invert_symbols(self, invert_symbols):
        self.invert_symbols = invert_symbols
        self.set_symbol_table(([1.0, -1.0] if self.invert_symbols else [-1.0, 1.0]))

    def get_baud(self):
        return self.baud

    def set_baud(self, baud):
        self.baud = baud
        self.set_baseband_rate(self.baud*self.sps)

    def get_symbol_table(self):
        return self.symbol_table

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table
        self.digital_chunks_to_symbols_bf_0.set_symbol_table(self.symbol_table)

    def get_sensitivity(self):
        return self.sensitivity

    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
        self.analog_frequency_modulator_fc_0.set_sensitivity(self.sensitivity)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.soapy_plutosdr_sink_0.set_sample_rate(0, self.samp_rate)

    def get_repeat(self):
        return self.repeat

    def set_repeat(self, repeat):
        self.repeat = repeat
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_preamble_flags(self):
        return self.preamble_flags

    def set_preamble_flags(self, preamble_flags):
        self.preamble_flags = preamble_flags
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_postamble_flags(self):
        return self.postamble_flags

    def set_postamble_flags(self, postamble_flags):
        self.postamble_flags = postamble_flags
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_phase_inc(self):
        return self.phase_inc

    def set_phase_inc(self, phase_inc):
        self.phase_inc = phase_inc
        self.blocks_rotator_cc_0.set_phase_inc(self.phase_inc)

    def get_payload_text(self):
        return self.payload_text

    def set_payload_text(self, payload_text):
        self.payload_text = payload_text
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_payload_mode(self):
        return self.payload_mode

    def set_payload_mode(self, payload_mode):
        self.payload_mode = payload_mode
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_payload_len(self):
        return self.payload_len

    def set_payload_len(self, payload_len):
        self.payload_len = payload_len
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_payload_hex(self):
        return self.payload_hex

    def set_payload_hex(self, payload_hex):
        self.payload_hex = payload_hex
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_output_path(self):
        return self.output_path

    def set_output_path(self, output_path):
        self.output_path = output_path
        self.blocks_file_sink_0.open(self.output_path)

    def get_nrzi_initial(self):
        return self.nrzi_initial

    def set_nrzi_initial(self, nrzi_initial):
        self.nrzi_initial = nrzi_initial
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_interp_out(self):
        return self.interp_out

    def set_interp_out(self, interp_out):
        self.interp_out = interp_out

    def get_gap_ms(self):
        return self.gap_ms

    def set_gap_ms(self, gap_ms):
        self.gap_ms = gap_ms
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.soapy_plutosdr_sink_0.set_frequency(0, self.freq)

    def get_decim_out(self):
        return self.decim_out

    def set_decim_out(self, decim_out):
        self.decim_out = decim_out

    def get_csp_src(self):
        return self.csp_src

    def set_csp_src(self, csp_src):
        self.csp_src = csp_src
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_sport(self):
        return self.csp_sport

    def set_csp_sport(self, csp_sport):
        self.csp_sport = csp_sport
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_priority(self):
        return self.csp_priority

    def set_csp_priority(self, csp_priority):
        self.csp_priority = csp_priority
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_flags(self):
        return self.csp_flags

    def set_csp_flags(self, csp_flags):
        self.csp_flags = csp_flags
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_dst(self):
        return self.csp_dst

    def set_csp_dst(self, csp_dst):
        self.csp_dst = csp_dst
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_dport(self):
        return self.csp_dport

    def set_csp_dport(self, csp_dport):
        self.csp_dport = csp_dport
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_data_text(self):
        return self.csp_data_text

    def set_csp_data_text(self, csp_data_text):
        self.csp_data_text = csp_data_text
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_data_hex(self):
        return self.csp_data_hex

    def set_csp_data_hex(self, csp_data_hex):
        self.csp_data_hex = csp_data_hex
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_crc_include_header(self):
        return self.csp_crc_include_header

    def set_csp_crc_include_header(self, csp_crc_include_header):
        self.csp_crc_include_header = csp_crc_include_header
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_csp_crc32c(self):
        return self.csp_crc32c

    def set_csp_crc32c(self, csp_crc32c):
        self.csp_crc32c = csp_crc32c
        self.blocks_vector_source_bits.set_data(ops_stage.make_bitstream(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])
        self.blocks_vector_source_gate.set_data(ops_stage.make_gate(self.repeat, self.payload_len, self.payload_mode, self.payload_text, self.payload_hex, self.preamble_flags, self.postamble_flags, self.gap_ms, self.nrzi_initial, self.sps, self.csp_data_text, self.csp_data_hex, self.csp_priority, self.csp_src, self.csp_dst, self.csp_dport, self.csp_sport, self.csp_flags, self.csp_crc32c, self.csp_crc_include_header), [])

    def get_bt(self):
        return self.bt

    def set_bt(self, bt):
        self.bt = bt
        self.interp_fir_filter_fff_0.set_taps(ops_stage.gmsk_taps(self.bt, self.sps))

    def get_baseband_rate(self):
        return self.baseband_rate

    def set_baseband_rate(self, baseband_rate):
        self.baseband_rate = baseband_rate

    def get_amplitude(self):
        return self.amplitude

    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
        self.blocks_multiply_const_amp.set_k(self.amplitude)




def main(top_block_cls=ops_sat_tx_wip, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
