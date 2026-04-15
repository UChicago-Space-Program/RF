"""
Quick spectrum scan to locate the Pluto TX signal relative to the RTL-SDR
center frequency. Captures 2 seconds of IQ at 1.152 MHz, computes an
averaged FFT, and prints the peak frequency offset.

Run with TX active and transmitting packets continuously.
"""

import sys
import time

import numpy as np

import osmosdr
from gnuradio import blocks, gr

CENTER_FREQ = 433_350_000
SAMP_RATE   = 1_152_000
RTL_GAIN    = 49.6
CAPTURE_SEC = 2.0
N_FFT       = 4096

class scan_fg(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        self.src = osmosdr.source(args="numchan=1 rtl=0")
        self.src.set_sample_rate(SAMP_RATE)
        self.src.set_center_freq(CENTER_FREQ, 0)
        self.src.set_freq_corr(0, 0)
        self.src.set_dc_offset_mode(0, 0)
        self.src.set_iq_balance_mode(0, 0)
        self.src.set_gain_mode(False, 0)
        self.src.set_gain(RTL_GAIN, 0)
        self.src.set_if_gain(20, 0)
        self.src.set_bb_gain(20, 0)
        self.src.set_antenna('', 0)

        n_samples = int(SAMP_RATE * CAPTURE_SEC)
        self.sink = blocks.vector_sink_c()
        self.head = blocks.head(gr.sizeof_gr_complex, n_samples)

        self.connect(self.src, self.head, self.sink)

fg = scan_fg()
print(f"Capturing {CAPTURE_SEC}s of IQ at {CENTER_FREQ/1e6:.3f} MHz, "
      f"samp_rate={SAMP_RATE/1e6:.3f} MHz ...")
fg.start()
fg.wait()

samples = np.array(fg.sink.data())
print(f"Captured {len(samples)} samples")

# Welch-style averaged FFT
n_avg = len(samples) // N_FFT
spectra = []
for i in range(n_avg):
    chunk = samples[i*N_FFT:(i+1)*N_FFT]
    window = np.hanning(N_FFT)
    spec = np.abs(np.fft.fftshift(np.fft.fft(chunk * window)))**2
    spectra.append(spec)

avg_spec = np.mean(spectra, axis=0)
avg_spec_db = 10 * np.log10(avg_spec + 1e-20)

freqs = np.fft.fftshift(np.fft.fftfreq(N_FFT, 1/SAMP_RATE))

# Find peak (exclude DC ±5 kHz to avoid DC spike artifact)
dc_mask = np.abs(freqs) < 5_000
avg_spec_db_masked = avg_spec_db.copy()
avg_spec_db_masked[dc_mask] = -200

peak_idx = np.argmax(avg_spec_db_masked)
peak_freq = freqs[peak_idx]
peak_power = avg_spec_db_masked[peak_idx]

noise_floor = np.median(avg_spec_db_masked[avg_spec_db_masked > -150])
snr = peak_power - noise_floor

print(f"\nPeak signal:")
print(f"  Offset from center : {peak_freq:+.0f} Hz ({peak_freq/1000:+.2f} kHz)")
print(f"  Absolute frequency : {(CENTER_FREQ + peak_freq)/1e6:.4f} MHz")
print(f"  Power              : {peak_power:.1f} dB (noise floor {noise_floor:.1f} dB)")
print(f"  SNR                : {snr:.1f} dB")
print()

if snr > 10:
    print(f"RECOMMENDATION: use --rx-offset {int(peak_freq)} in rtlsdr_rx.py")
    print(f"  python -u rtlsdr_rx.py --rx-offset {int(peak_freq)}")
else:
    print("WARNING: SNR is low (<10 dB). TX may not be transmitting, or signal is")
    print("very weak. Try sending packets from the test script while this runs.")

# Also print top 5 peaks for context
print("\nTop 5 peaks (excluding DC):")
top5 = np.argsort(avg_spec_db_masked)[-5:][::-1]
for idx in top5:
    print(f"  {freqs[idx]:+8.0f} Hz  ({avg_spec_db_masked[idx]:.1f} dB)")
