#!/usr/bin/env python3
"""
2-D sweep: TX attenuation × Doppler offset → FER
================================================
Runs a trial (500 packets) for every (attn, freq_offset) combination and
saves results to doppler_attn_results.csv.  Generates a 3-D surface plot
(doppler_attn_surface.png) when complete.

Grid
----
  ATTN_VALUES     : 6 attenuation levels spanning clean → failure
  FREQ_OFFSETS_HZ : ±5 kHz in 1 kHz steps (11 points)
  Total trials    : 66  (~63 min at ~57 s/trial)

Hardware notes
--------------
* Requires rtlsdr_rx.py and plutosdr_tx.py already running before launch.
* TX LO is shifted via iio_attr (altvoltage1 frequency).
* RX flowgraph is NOT retuned — the test measures receiver Doppler tolerance.
* FSK deviation is 3 kHz, so ±3 kHz offsets are at the demodulator edge;
  beyond ±4–5 kHz the binary slicer collapses and FER → 1.
"""

import atexit
import csv
import random
import signal
import socket
import struct
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TX_HOST, TX_PORT           = '127.0.0.1', 52001
RX_BIND_HOST, RX_BIND_PORT = '127.0.0.1', 52002
PLUTO_URI                  = 'ip:192.168.2.1'
NOMINAL_FREQ_HZ            = 433_350_000

ATTN_VALUES     = [0, 35, 42, 46, 50, 52]          # dB — clean, cliff, failure
FREQ_OFFSETS_HZ = [-5000, -4000, -3000, -2000, -1000,
                    0,
                    1000,  2000,  3000,  4000,  5000]  # Hz

NUM_PACKETS         = 500
PAYLOAD_LEN         = 60
INTER_PACKET_DELAY  = 0.05   # s between TX UDP sends
RX_DRAIN_TIME       = 30.0   # s — must exceed (500*88ms - 25s) ≈ 19 s
ATTN_SETTLE_TIME    = 1.5    # s after changing attenuation
FREQ_SETTLE_TIME    = 2.0    # s after changing TX LO (AD9361 PLL + RX reacquire)

CSV_PATH    = 'doppler_attn_results.csv'
PLOT_PATH   = 'doppler_attn_surface.png'


# ---------------------------------------------------------------------------
# Hardware control
# ---------------------------------------------------------------------------

def set_pluto_attn(db: float):
    subprocess.run(
        ['iio_attr', '-u', PLUTO_URI, '-c', 'ad9361-phy',
         '-o', 'voltage0', 'hardwaregain', str(-abs(db))],
        check=True, capture_output=True)


def set_pluto_tx_freq(hz: int):
    subprocess.run(
        ['iio_attr', '-u', PLUTO_URI, '-c', 'ad9361-phy',
         'altvoltage1', 'frequency', str(int(hz))],
        check=True, capture_output=True)


def restore_hardware():
    """Reset Pluto to nominal state — called on exit / Ctrl-C."""
    try:
        set_pluto_tx_freq(NOMINAL_FREQ_HZ)
        set_pluto_attn(0)
        print('\n[cleanup] Pluto restored: freq=nominal, attn=0 dB', flush=True)
    except Exception as e:
        print(f'\n[cleanup] Warning: could not restore Pluto: {e}', flush=True)


atexit.register(restore_hardware)
signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# ---------------------------------------------------------------------------
# AX.25 frame helpers
# ---------------------------------------------------------------------------

def _encode_addr(callsign, ssid, last):
    cs = (callsign.upper() + '      ')[:6]
    out = bytearray(c << 1 for c in cs.encode('ascii'))
    out.append(0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0))
    return bytes(out)


HDR = (_encode_addr('DEST  ', 0, False) +
       _encode_addr('SRC   ', 0, True) +
       bytes([0x03, 0xF0]))

_POPCOUNT = bytes(bin(i).count('1') for i in range(256))


def payload_for(seq):
    rng = random.Random(seq)
    return bytes(rng.getrandbits(8) for _ in range(PAYLOAD_LEN))


def build_pdu(seq):
    return HDR + struct.pack('>I', seq) + payload_for(seq)


def parse_received(data):
    HLEN = 16
    if len(data) < HLEN + 4:
        return None, None
    body = data[HLEN:]
    return struct.unpack('>I', body[:4])[0], body[4:4 + PAYLOAD_LEN]


def popcount_xor(a, b):
    n = min(len(a), len(b))
    return sum(_POPCOUNT[a[i] ^ b[i]] for i in range(n)) + abs(len(a) - len(b)) * 8


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(label, attn_db, freq_offset_hz):
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((RX_BIND_HOST, RX_BIND_PORT))
    rx.settimeout(0.05)
    received = {}

    print(f'  [{label}] Sending {NUM_PACKETS} PDUs ...', flush=True)
    for seq in range(NUM_PACKETS):
        tx.sendto(build_pdu(seq), (TX_HOST, TX_PORT))
        try:
            data, _ = rx.recvfrom(4096)
            s, p = parse_received(data)
            if s is not None and 0 <= s < NUM_PACKETS:
                received[s] = p
        except socket.timeout:
            pass
        time.sleep(INTER_PACKET_DELAY)

    print(f'  [{label}] Draining {RX_DRAIN_TIME:.0f}s ...', flush=True)
    deadline = time.time() + RX_DRAIN_TIME
    while time.time() < deadline:
        try:
            data, _ = rx.recvfrom(4096)
            s, p = parse_received(data)
            if s is not None and 0 <= s < NUM_PACKETS:
                received[s] = p
        except socket.timeout:
            pass

    tx.close()
    rx.close()

    total_bits   = NUM_PACKETS * PAYLOAD_LEN * 8
    total_errors = frame_errors = frames_missing = 0
    for seq in range(NUM_PACKETS):
        expected = payload_for(seq)
        got = received.get(seq)
        if got is None:
            frames_missing += 1
            frame_errors   += 1
            total_errors   += PAYLOAD_LEN * 8
        else:
            be = popcount_xor(expected, got)
            if be > 0:
                frame_errors += 1
            total_errors += be

    fer = frame_errors / NUM_PACKETS
    ber = total_errors / total_bits
    print(f'  [{label}] recv={len(received):3d}/{NUM_PACKETS}  '
          f'missing={frames_missing:3d}  FER={fer:.4f}  BER={ber:.6f}', flush=True)

    return {
        'label':          label,
        'attn_db':        attn_db,
        'freq_offset_hz': freq_offset_hz,
        'sent':           NUM_PACKETS,
        'received':       len(received),
        'missing':        frames_missing,
        'frame_errors':   frame_errors,
        'fer':            fer,
        'ber':            ber,
    }


# ---------------------------------------------------------------------------
# 3-D surface plot
# ---------------------------------------------------------------------------

def plot_surface(results):
    attns   = sorted(set(r['attn_db']        for r in results))
    offsets = sorted(set(r['freq_offset_hz'] for r in results))

    # Build FER grid  shape: (n_offsets, n_attns)
    fer_grid = np.zeros((len(offsets), len(attns)))
    for r in results:
        i = offsets.index(r['freq_offset_hz'])
        j = attns.index(r['attn_db'])
        fer_grid[i, j] = r['fer']

    X, Y = np.meshgrid(attns, [o / 1000 for o in offsets])  # Y in kHz

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, fer_grid,
                           cmap='RdYlGn_r',
                           alpha=0.92,
                           linewidth=0.3,
                           edgecolor='#555555',
                           antialiased=True)

    ax.set_xlabel('TX attenuation (dB)', labelpad=10)
    ax.set_ylabel('Doppler offset (kHz)', labelpad=10)
    ax.set_zlabel('FER', labelpad=8)
    ax.set_zlim(0, 1)
    ax.set_title('Frame Error Rate — TX Attenuation × Doppler Offset\n'
                 'PlutoSDR → RTL-SDR, GFSK 9600 baud, 433 MHz',
                 fontsize=12, fontweight='bold', pad=15)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label('FER', labelpad=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.view_init(elev=28, azim=-55)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180, bbox_inches='tight')
    print(f'Saved {PLOT_PATH}', flush=True)
    plt.close()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

n_total  = len(ATTN_VALUES) * len(FREQ_OFFSETS_HZ)
n_done   = 0
results  = []

print(f'=== 2-D Doppler × Attenuation sweep ===', flush=True)
print(f'    {len(ATTN_VALUES)} attn values × {len(FREQ_OFFSETS_HZ)} freq offsets '
      f'= {n_total} trials', flush=True)
print(f'    Estimated time: ~{n_total * 57 // 60} min\n', flush=True)

for attn in ATTN_VALUES:
    print(f'\n--- Attn = {attn} dB ---', flush=True)
    set_pluto_attn(attn)
    time.sleep(ATTN_SETTLE_TIME)

    for offset in FREQ_OFFSETS_HZ:
        n_done += 1
        label = f'{attn}dB / {offset:+d}Hz'
        print(f'\n[{n_done}/{n_total}] {label}', flush=True)

        set_pluto_tx_freq(NOMINAL_FREQ_HZ + offset)
        time.sleep(FREQ_SETTLE_TIME)

        results.append(run_trial(label, attn, offset))

    # Reset frequency to nominal before moving to next attenuation
    set_pluto_tx_freq(NOMINAL_FREQ_HZ)
    time.sleep(0.5)

# Restore hardware (atexit also covers crash/Ctrl-C)
restore_hardware()

# Print summary table
print('\n=== RESULTS SUMMARY ===', flush=True)
print(f'{"Attn":>6}  {"Offset Hz":>10}  {"recv":>6}  {"FER":>8}', flush=True)
for r in results:
    print(f"{r['attn_db']:>6}  {r['freq_offset_hz']:>10}  "
          f"{r['received']:>6}/{r['sent']}  {r['fer']:>8.4f}", flush=True)

# Save CSV
with open(CSV_PATH, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'label', 'attn_db', 'freq_offset_hz', 'sent', 'received',
        'missing', 'frame_errors', 'fer', 'ber'])
    w.writeheader()
    w.writerows(results)
print(f'\nWrote {CSV_PATH}', flush=True)

# Generate 3-D surface plot
plot_surface(results)
