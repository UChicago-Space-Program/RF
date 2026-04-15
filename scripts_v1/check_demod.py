"""
Diagnostic: capture 3s of IQ, apply freq_xlating + quadrature demod,
print statistics on the demod output to verify signal levels look right.
"""

import math
import sys
import numpy as np
import osmosdr
from gnuradio import analog, blocks, filter, gr
from gnuradio.filter import firdes
from gnuradio.fft import window

CENTER_FREQ     = 433_350_000
SAMP_RATE_IN    = 1_152_000
DECIM           = 2
SAMP_RATE       = SAMP_RATE_IN // DECIM
RTL_GAIN        = 40
FSK_DEV_HZ      = 3000
CAPTURE_SEC     = 3.0
RX_OFFSET_HZ    = int(sys.argv[1]) if len(sys.argv) > 1 else -5062

n_samples_in  = int(SAMP_RATE_IN * CAPTURE_SEC)
n_samples_out = n_samples_in // DECIM

class demod_fg(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        self.src = osmosdr.source(args="numchan=1 rtl=0")
        self.src.set_sample_rate(SAMP_RATE_IN)
        self.src.set_center_freq(CENTER_FREQ, 0)
        self.src.set_freq_corr(0, 0)
        self.src.set_dc_offset_mode(0, 0)
        self.src.set_iq_balance_mode(0, 0)
        self.src.set_gain_mode(False, 0)
        self.src.set_gain(RTL_GAIN, 0)
        self.src.set_if_gain(20, 0)
        self.src.set_bb_gain(20, 0)

        self.xlate = filter.freq_xlating_fir_filter_ccf(
            DECIM,
            firdes.low_pass(1.0, SAMP_RATE_IN, 25_000, 8_000),
            RX_OFFSET_HZ,
            SAMP_RATE_IN)

        self.demod = analog.quadrature_demod_cf(
            SAMP_RATE / (2 * math.pi * FSK_DEV_HZ))

        self.lp = filter.fir_filter_fff(
            1,
            firdes.low_pass(1, SAMP_RATE, 15_000, 5_000,
                            window.WIN_HAMMING, 6.76))

        # Sink captures demod output (after LPF) as floats
        self.sink = blocks.vector_sink_f()
        self.head = blocks.head(gr.sizeof_float, n_samples_out)

        self.connect(self.src, self.xlate, self.demod, self.lp, self.head, self.sink)

print(f"Capturing {CAPTURE_SEC}s with rx_offset={RX_OFFSET_HZ} Hz ...")
fg = demod_fg()
fg.start()
fg.wait()

data = np.array(fg.sink.data())
print(f"Demod output: {len(data)} samples")
if len(data) == 0:
    print("ERROR: no samples captured")
    sys.exit(1)

print(f"  Mean   : {data.mean():.4f}  (expect ~0 if carrier centered)")
print(f"  Std dev: {data.std():.4f}   (expect ~0.6-1.2 for GFSK ±1 swing)")
print(f"  Min    : {data.min():.4f}")
print(f"  Max    : {data.max():.4f}")
print(f"  Range  : {data.max()-data.min():.4f}  (expect ~2.0 for ±1 swing)")

# histogram of values to see bimodal (mark/space) distribution
bins = np.linspace(-4, 4, 81)
hist, _ = np.histogram(data, bins=bins)
print("\nHistogram of demod output (ideal: peaks near -1 and +1):")
peak_bins = np.argsort(hist)[-5:][::-1]
for b in sorted(peak_bins):
    bar = '#' * (hist[b] // max(hist.max() // 40, 1))
    print(f"  {bins[b]:+.2f} to {bins[b+0.05]:.2f}: {bar} ({hist[b]})")

# Estimate carrier offset from mean
carrier_offset_hz = data.mean() * FSK_DEV_HZ
print(f"\nEstimated residual carrier offset: {carrier_offset_hz:.0f} Hz")
if abs(carrier_offset_hz) > 1000:
    suggested = RX_OFFSET_HZ - int(carrier_offset_hz)
    print(f"  Carrier not centered! Suggested rx_offset: {suggested}")
else:
    print(f"  Carrier looks well-centered.")
