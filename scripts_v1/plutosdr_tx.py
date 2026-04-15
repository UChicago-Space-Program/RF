#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PlutoSDR TX flowgraph for the BER/FER test harness.
#
# Differences from the original Plutosdr_tx_sample.py:
#   - Headless (no Qt). The TX side has nothing to display.
#   - `pdu.random_pdu` removed. Data now comes from a UDP socket_pdu.
#   - `network.socket_pdu` is in UDP_SERVER mode on 127.0.0.1:52001 and
#     wired to the HDLC framer's input. ber_fer_test.py sends each
#     frame as one UDP datagram, which becomes one PDU here.
#   - samp_rate fixed from 567000 -> 576000 (the original was a typo:
#     60 sps * 9600 baud = 576000 Hz, not 567000).
#
# Bug fixes applied (all were in the original .py, not the .grc):
#   1. unpack_k_bits_bb REMOVED: hdlc_framer already outputs individual bits
#      (0/1 per byte in the PDU u8vector). The block was expanding each bit
#      into 8 bits, transmitting at 1/8 speed with 7/8 zero bits.
#   2. nrzi_encode ADDED: ax25_deframer on the RX side runs nrzi_decode()
#      as part of its chain. TX must NRZ-I encode the bit stream or the RX
#      will see inverted transitions and decode garbage.
#   3. sensitivity fixed: was 1.0 rad/sample → ~91 kHz deviation, 30× wider
#      than the RX's freq_xlating LPF (25 kHz). Now computed from
#      fsk_deviation_hz=3000 to match the quadrature_demod gain in
#      rtlsdr_rx.py: sensitivity = 2π × 3000 / 576000 ≈ 0.03272 rad/sample.
#   4. do_unpack=False: hdlc_framer outputs unpacked bits, so gfsk_mod must
#      not do its own internal unpack (which would re-interpret 0/1 bytes as
#      packed bytes and strip all but their MSB, giving all zeros).
#   5. band_pass(1k, 100k) → low_pass(100k): the bandpass had a DC notch
#      that distorted the baseband GFSK signal (occupies ±3 kHz) during bit
#      transitions when the instantaneous frequency crosses 0 Hz.
#
# The frames sent over UDP are expected to already contain a valid
# 16-byte AX.25 UI header (built by the test script) followed by the
# info field. The HDLC framer wraps the whole thing with flags + bit
# stuffing + CRC, GFSK-modulates, and pushes IQ to the Pluto.

import signal
import sys
import threading
from argparse import ArgumentParser

from gnuradio import blocks, digital, filter, gr, iio, network, pdu
from gnuradio.eng_arg import eng_float
from gnuradio.fft import window
from gnuradio.filter import firdes
import satellites


class plutosdr_tx(gr.top_block):
    def __init__(self,
                 samp_rate=576000,
                 center_frequency=433_350_000,
                 tx_attenuation_db=10.0,
                 udp_host="127.0.0.1",
                 udp_port=52001):
        gr.top_block.__init__(self, "PlutoSDR TX (BER harness)",
                              catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        # ---------------- Variables ----------------
        self.samp_rate = samp_rate
        self.center_frequency = center_frequency
        self.tx_attenuation_db = tx_attenuation_db

        # ---------------- Blocks -------------------
        # Data ingress: each UDP datagram from the BER script becomes one PDU.
        self.network_socket_pdu_0 = network.socket_pdu(
            'UDP_SERVER', udp_host, str(udp_port), 10000, False)

        # The PDU is already a complete AX.25 frame (header + info), so we
        # just need to HDLC-frame it for the air.
        # hdlc_framer outputs a PDU of INDIVIDUAL BITS (0/1 per byte as u8vector),
        # not packed bytes. The chain downstream must not re-unpack these.
        self.satellites_hdlc_framer_0 = satellites.hdlc_framer(
            preamble_bytes=50, postamble_bytes=7)

        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(
            gr.types.byte_t, 'packet_len')

        # NRZ-I encode the bit stream BEFORE GFSK modulation.
        # ax25_deframer on the RX runs nrzi_decode() as part of its chain,
        # so the TX must encode: NRZ 0 → frequency transition, NRZ 1 → no transition.
        self.satellites_nrzi_encode_0 = satellites.nrzi_encode()

        # sensitivity = 2π × fsk_deviation_hz / samp_rate
        # Must match the quadrature_demod gain in rtlsdr_rx.py (fsk_deviation_hz=3000).
        # Original sensitivity=1.0 → ~91 kHz deviation — 30× wider than the RX filters.
        gfsk_sensitivity = 2 * 3.141592653589793 * 3000 / samp_rate  # ≈ 0.03272 rad/sample
        self.digital_gfsk_mod_0 = digital.gfsk_mod(
            samples_per_symbol=60,
            sensitivity=gfsk_sensitivity,
            bt=0.35,
            verbose=False,
            log=False,
            do_unpack=False)   # hdlc_framer output is already one bit per byte

        # Low-pass filter to limit out-of-band emissions before the Pluto DAC.
        # (Original band_pass(1k, 100k) had a DC notch that distorted the GFSK
        # signal whenever the instantaneous frequency crossed through 0 Hz.)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1, samp_rate, 100_000, 20_000,
                window.WIN_HAMMING, 6.76))

        self.iio_pluto_sink_0 = iio.fmcomms2_sink_fc32(
            iio.get_pluto_uri(), [True, True], 32768, False)
        self.iio_pluto_sink_0.set_len_tag_key('')
        self.iio_pluto_sink_0.set_bandwidth(20_000_000)
        self.iio_pluto_sink_0.set_frequency(center_frequency)
        self.iio_pluto_sink_0.set_samplerate(samp_rate)
        self.iio_pluto_sink_0.set_attenuation(0, tx_attenuation_db)
        self.iio_pluto_sink_0.set_filter_params('Auto', '', 0, 0)

        # ---------------- Connections --------------
        # Message path: UDP -> HDLC framer -> tagged stream
        self.msg_connect((self.network_socket_pdu_0, 'pdus'),
                         (self.satellites_hdlc_framer_0, 'in'))
        self.msg_connect((self.satellites_hdlc_framer_0, 'out'),
                         (self.pdu_pdu_to_tagged_stream_0, 'pdus'))

        # Stream path: bits → NRZI encode → GFSK → LPF → Pluto
        # Note: no unpack_k_bits_bb — hdlc_framer already outputs one bit per byte.
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0),
                     (self.satellites_nrzi_encode_0, 0))
        self.connect((self.satellites_nrzi_encode_0, 0),
                     (self.digital_gfsk_mod_0, 0))
        self.connect((self.digital_gfsk_mod_0, 0),
                     (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0),
                     (self.iio_pluto_sink_0, 0))

    # Setters for live attenuation/frequency changes (used by ber_fer_test_auto.py)
    def set_tx_attenuation_db(self, db):
        self.tx_attenuation_db = db
        self.iio_pluto_sink_0.set_attenuation(0, db)

    def set_center_frequency(self, hz):
        self.center_frequency = hz
        self.iio_pluto_sink_0.set_frequency(hz)


def main():
    parser = ArgumentParser()
    parser.add_argument("--samp-rate", type=eng_float, default=576_000)
    parser.add_argument("--freq", type=eng_float, default=433_350_000)
    parser.add_argument("--attn", type=float, default=10.0,
                        help="Pluto TX attenuation in dB (0..89.75)")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=52001)
    args = parser.parse_args()

    tb = plutosdr_tx(
        samp_rate=int(args.samp_rate),
        center_frequency=int(args.freq),
        tx_attenuation_db=args.attn,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )

    def sig_handler(*_):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()
    print(f"[plutosdr_tx] running. UDP {args.udp_host}:{args.udp_port} -> "
          f"HDLC -> NRZI -> GFSK -> Pluto @ {args.freq/1e6:.3f} MHz, "
          f"attn={args.attn} dB. Ctrl-C to quit.")
    tb.wait()


if __name__ == "__main__":
    main()
