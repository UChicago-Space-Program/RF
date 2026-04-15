#!/usr/bin/env python3
"""
Publication sweep: 500 packets per trial, fine steps bracketing the FER cliff
(cliff location varies 40–55 dB with indoor multipath; run a fresh flowgraph before starting).
"""
import csv, random, socket, struct, subprocess, sys, time
import matplotlib.pyplot as plt

TX_HOST, TX_PORT = '127.0.0.1', 52001
RX_BIND_HOST, RX_BIND_PORT = '127.0.0.1', 52002
PLUTO_URI = 'ip:192.168.2.1'

# Fine sweep from well below to well above the cliff (cliff is at ~45–50 dB)
ATTN_VALUES = [0, 10, 20, 30, 35, 40, 42, 44, 45, 46, 47, 48, 49, 50, 52, 55, 60]

NUM_PACKETS = 500
PAYLOAD_LEN = 60
INTER_PACKET_DELAY = 0.05
RX_DRAIN_TIME = 30.0   # must be > (500 pkts * ~88ms/frame - 25s send phase) = ~19s; 30s is safe
SETTLE_TIME = 1.5

CSV_PATH  = 'ber_results.csv'
PLOT_PATH     = 'ber_fer_plot.png'
PLOT_PATH_LIN = 'ber_fer_plot_linear.png'


def encode_ax25_address(callsign, ssid, last):
    cs = (callsign.upper() + '      ')[:6]
    out = bytearray(c << 1 for c in cs.encode('ascii'))
    out.append(0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0))
    return bytes(out)

HDR = (encode_ax25_address('DEST  ', 0, False) +
       encode_ax25_address('SRC   ', 0, True) + bytes([0x03, 0xF0]))
_POPCOUNT = bytes(bin(i).count('1') for i in range(256))

def popcount_xor(a, b):
    n = min(len(a), len(b))
    return sum(_POPCOUNT[a[i] ^ b[i]] for i in range(n)) + abs(len(a)-len(b))*8

def payload_for(seq):
    rng = random.Random(seq)
    return bytes(rng.getrandbits(8) for _ in range(PAYLOAD_LEN))

def build_pdu(seq):
    return HDR + struct.pack('>I', seq) + payload_for(seq)

def parse_received(data):
    HLEN = 16
    if len(data) < HLEN + 4: return None, None
    body = data[HLEN:]
    return struct.unpack('>I', body[:4])[0], body[4:4+PAYLOAD_LEN]

def set_pluto_attn(db):
    subprocess.run(['iio_attr', '-u', PLUTO_URI, '-c', 'ad9361-phy',
                    '-o', 'voltage0', 'hardwaregain', str(-abs(db))],
                   check=True, capture_output=True)

def run_trial(label, attn_db):
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((RX_BIND_HOST, RX_BIND_PORT))
    rx.settimeout(0.05)
    received = {}

    print(f'[{label}] Sending {NUM_PACKETS} PDUs ...', flush=True)
    for seq in range(NUM_PACKETS):
        tx.sendto(build_pdu(seq), (TX_HOST, TX_PORT))
        try:
            data, _ = rx.recvfrom(4096)
            s, p = parse_received(data)
            if s is not None and 0 <= s < NUM_PACKETS: received[s] = p
        except socket.timeout: pass
        time.sleep(INTER_PACKET_DELAY)

    print(f'[{label}] Draining {RX_DRAIN_TIME:.0f}s ...', flush=True)
    deadline = time.time() + RX_DRAIN_TIME
    while time.time() < deadline:
        try:
            data, _ = rx.recvfrom(4096)
            s, p = parse_received(data)
            if s is not None and 0 <= s < NUM_PACKETS: received[s] = p
        except socket.timeout: pass

    tx.close(); rx.close()
    total_bits = NUM_PACKETS * PAYLOAD_LEN * 8
    total_errors = frame_errors = frames_missing = 0
    for seq in range(NUM_PACKETS):
        expected = payload_for(seq)
        got = received.get(seq)
        if got is None:
            frames_missing += 1; frame_errors += 1; total_errors += PAYLOAD_LEN * 8
        else:
            be = popcount_xor(expected, got)
            if be > 0: frame_errors += 1
            total_errors += be
    fer = frame_errors / NUM_PACKETS
    ber = total_errors / total_bits
    print(f'[{label}] recv={len(received):3d}/{NUM_PACKETS} '
          f'missing={frames_missing:3d} FER={fer:.4f} BER={ber:.6f}', flush=True)
    return {'label': label, 'attn_db': attn_db, 'sent': NUM_PACKETS,
            'received': len(received), 'missing': frames_missing,
            'frame_errors': frame_errors, 'fer': fer, 'ber': ber}


TITLE = ('BER / FER vs TX Attenuation\n'
         'PlutoSDR TX → RTL-SDR Blog V4 RX, AX.25 UI, GFSK 9600 baud, '
         '433.35 MHz, over-the-air loopback ~50 cm')


def plot_results(results):
    attns  = [r['attn_db'] for r in results]

    floor_ber = 0.5 / (NUM_PACKETS * PAYLOAD_LEN * 8)
    floor_fer = 0.5 / NUM_PACKETS
    bers_floor = [max(r['ber'], floor_ber) for r in results]
    fers_floor = [max(r['fer'], floor_fer) for r in results]
    fers_raw   = [r['fer'] for r in results]
    bers_raw   = [r['ber'] for r in results]

    # --- log-scale plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(attns, fers_floor, marker='o', label='FER')
    ax.semilogy(attns, bers_floor, marker='s', linestyle='--', label='BER (≈ FER for CRC-protected)')
    ax.set_xlabel('TX attenuation (dB)')
    ax.set_ylabel('Error rate (log scale)')
    ax.set_title(TITLE)
    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    ax.set_xticks(attns)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f'Saved {PLOT_PATH}', flush=True)
    plt.close()

    # --- linear-scale plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(attns, fers_raw, marker='o', label='FER')
    ax.plot(attns, bers_raw, marker='s', linestyle='--', label='BER (≈ FER for CRC-protected)')
    ax.set_xlabel('TX attenuation (dB)')
    ax.set_ylabel('Error rate (linear)')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(TITLE)
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.set_xticks(attns)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_PATH_LIN, dpi=150)
    print(f'Saved {PLOT_PATH_LIN}', flush=True)
    plt.close()


results = []
for attn in ATTN_VALUES:
    label = f'{attn} dB'
    print(f'\n--- Pluto attn -> {attn} dB ---', flush=True)
    set_pluto_attn(attn)
    time.sleep(SETTLE_TIME)
    results.append(run_trial(label, attn))

# Reset attn
set_pluto_attn(0)

print('\n=== PUBLICATION SWEEP RESULTS ===', flush=True)
for r in results:
    print(f"  {r['label']:10s}  recv={r['received']:3d}/{r['sent']}  "
          f"FER={r['fer']:.4f}  BER={r['ber']:.6f}", flush=True)

with open(CSV_PATH, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['label','attn_db','sent','received',
                                      'missing','frame_errors','fer','ber'])
    w.writeheader(); w.writerows(results)
print(f'Wrote {CSV_PATH}', flush=True)

plot_results(results)
