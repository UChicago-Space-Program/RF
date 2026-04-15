#!/usr/bin/env python3
"""Probe sweep: 8 attenuation steps, 100 packets each. Find the FER cliff."""
import csv, random, socket, struct, subprocess, sys, time

TX_HOST, TX_PORT = '127.0.0.1', 52001
RX_BIND_HOST, RX_BIND_PORT = '127.0.0.1', 52002
PLUTO_URI = 'ip:192.168.2.1'
ATTN_VALUES = [0, 10, 20, 30, 40, 50, 60, 70]
NUM_PACKETS = 100
PAYLOAD_LEN = 60
INTER_PACKET_DELAY = 0.05
RX_DRAIN_TIME = 15.0

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

results = []
for attn in ATTN_VALUES:
    label = f'{attn} dB'
    print(f'\n--- Pluto attn -> {attn} dB ---', flush=True)
    set_pluto_attn(attn)
    time.sleep(1.5)
    results.append(run_trial(label, attn))

# Reset attn to 0 when done
set_pluto_attn(0)

print('\n=== PROBE SWEEP RESULTS ===', flush=True)
for r in results:
    print(f"  {r['label']:10s}  recv={r['received']:3d}/{r['sent']}  "
          f"FER={r['fer']:.4f}  BER={r['ber']:.6f}", flush=True)

with open('probe_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['label','attn_db','sent','received',
                                      'missing','frame_errors','fer','ber'])
    w.writeheader(); w.writerows(results)
print('Wrote probe_results.csv', flush=True)
