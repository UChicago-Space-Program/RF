"""
BER / FER test harness for the PlutoSDR -> RTL-SDR + AX.25 loopback.

How this fits with the GNU Radio flowgraphs
-------------------------------------------
Run all three pieces in this order:

  1. python rtlsdr_rx.py    (RTL-SDR receiver, opens a Qt window)
  2. python plutosdr_tx.py  (PlutoSDR transmitter, headless)
  3. python ber_fer_test.py (this script)

The two flowgraphs each expose a `network.socket_pdu` block in UDP
mode. This script:
  - Sends UDP datagrams to the TX flowgraph at 127.0.0.1:52001.
    Each datagram is one complete AX.25 frame (header + info field).
    The TX flowgraph HDLC-frames it and modulates it onto the air.
  - Listens for UDP datagrams on 127.0.0.1:52002. The RX flowgraph's
    AX.25 deframer drops everything that fails CRC, so each datagram
    we receive is a bit-perfect AX.25 PDU.

Why we build the AX.25 header in Python
---------------------------------------
gr-satellites has an `hdlc_framer` and an `ax25_deframer`, but no
`ax25_framer`. Per the gr-satellites discussions, the standard pattern
is to build the AX.25 header (callsigns + control + PID, 16 bytes for
a UI frame) externally and feed the full pre-framed AX.25 packet into
the HDLC framer as bytes. So this script constructs the header here.

Frame layout (each UDP datagram, sent and received):

    [ AX.25 HEADER 16 B ][ SEQ 4 B BE uint ][ PAYLOAD PAYLOAD_LEN B ]

The AX.25 header is constant across all frames in a run. The payload
for sequence N is `random.Random(N).getrandbits(8) * PAYLOAD_LEN`,
so the receiver can recompute the expected bytes for any seq without
TX state.

What this measures
------------------
Because the AX.25 deframer enforces a CRC, every PDU we ever see on
the RX side is bit-perfect. So in practice the BER curve will track
the FER curve almost exactly (BER = FER, near enough). This is normal
for any FCS-protected link — frame loss IS the failure mode. The
script still prints BER alongside FER so the CSV looks complete.

If you want a real uncoded BER curve later, that requires tapping the
chain *before* the deframer (e.g. off the slicer), which is a separate
experiment.

Outputs
-------
- ber_results.csv   one row per attenuation setting
- ber_fer_plot.png  log-scale plot of BER and FER vs trial label
"""

import csv
import random
import socket
import struct
import time

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration — edit to taste.
# ---------------------------------------------------------------------------

TX_HOST = "127.0.0.1"
TX_PORT = 52001          # plutosdr_tx.py socket_pdu UDP_SERVER

RX_BIND_HOST = "127.0.0.1"
RX_BIND_PORT = 52002     # we listen here; rtlsdr_rx.py UDP_CLIENTs to us

NUM_PACKETS = 500
PAYLOAD_LEN = 60         # 4 (seq) + 60 (payload) = 64-byte info field

INTER_PACKET_DELAY = 0.05   # seconds; 9600 baud is slow, give it room
RX_DRAIN_TIME      = 8.0    # how long to wait for stragglers after last TX
RX_RECV_TIMEOUT    = 0.2

# AX.25 callsigns. Anything six chars + numeric SSID 0-15 works.
DEST_CALL = "DEST  "
DEST_SSID = 0
SRC_CALL  = "SRC   "
SRC_SSID  = 0

TRIAL_LABELS = [
    "0 dB attn",
    "10 dB attn",
    "20 dB attn",
    "30 dB attn",
]

CSV_PATH  = "ber_results.csv"
PLOT_PATH = "ber_fer_plot.png"


# ---------------------------------------------------------------------------
# AX.25 header construction
# ---------------------------------------------------------------------------

def encode_ax25_address(callsign: str, ssid: int, last: bool) -> bytes:
    """Encode one 7-byte AX.25 address field.

    AX.25 left-shifts every byte of the address field by 1 bit. The
    last byte of the source address has its LSB set to mark the end
    of the address sequence.
    """
    cs = (callsign.upper() + "      ")[:6]   # space-pad to 6 chars
    out = bytearray(c << 1 for c in cs.encode("ascii"))
    # SSID byte: 0b011SSSSE  (the 0x60 sets two reserved-1 bits)
    out.append(0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0))
    return bytes(out)


def build_ax25_header() -> bytes:
    """16-byte AX.25 UI frame header: dest + src + control + PID."""
    dest = encode_ax25_address(DEST_CALL, DEST_SSID, last=False)
    src  = encode_ax25_address(SRC_CALL,  SRC_SSID,  last=True)
    control = bytes([0x03])   # UI frame, no P/F
    pid     = bytes([0xF0])   # no layer-3 protocol
    return dest + src + control + pid


AX25_HEADER = build_ax25_header()
assert len(AX25_HEADER) == 16, "AX.25 header should be exactly 16 bytes"
HEADER_LEN = len(AX25_HEADER)


# ---------------------------------------------------------------------------
# Payload + frame helpers
# ---------------------------------------------------------------------------

# 256-byte popcount table — works on every Python 3, unlike int.bit_count()
# which is 3.10+ only.
_POPCOUNT = bytes(bin(i).count("1") for i in range(256))


def popcount_xor(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    diff = 0
    for i in range(n):
        diff += _POPCOUNT[a[i] ^ b[i]]
    diff += abs(len(a) - len(b)) * 8
    return diff


def payload_for(seq: int) -> bytes:
    rng = random.Random(seq)
    return bytes(rng.getrandbits(8) for _ in range(PAYLOAD_LEN))


def build_pdu(seq: int) -> bytes:
    return AX25_HEADER + struct.pack(">I", seq) + payload_for(seq)


def parse_received(data: bytes):
    """Strip the AX.25 header, return (seq, payload) or (None, None)."""
    if len(data) < HEADER_LEN + 4:
        return None, None
    body = data[HEADER_LEN:]
    seq = struct.unpack(">I", body[:4])[0]
    payload = body[4 : 4 + PAYLOAD_LEN]
    return seq, payload


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_trial(label: str) -> dict:
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((RX_BIND_HOST, RX_BIND_PORT))
    rx.settimeout(RX_RECV_TIMEOUT)

    received = {}

    print(f"\n[{label}] Sending {NUM_PACKETS} PDUs to {TX_HOST}:{TX_PORT} ...")
    for seq in range(NUM_PACKETS):
        tx.sendto(build_pdu(seq), (TX_HOST, TX_PORT))
        if INTER_PACKET_DELAY:
            time.sleep(INTER_PACKET_DELAY)

    print(f"[{label}] Draining RX on {RX_BIND_HOST}:{RX_BIND_PORT} "
          f"for {RX_DRAIN_TIME:.1f} s ...")
    deadline = time.time() + RX_DRAIN_TIME
    while time.time() < deadline:
        try:
            data, _ = rx.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break

        seq, payload = parse_received(data)
        if seq is None:
            continue
        if 0 <= seq < NUM_PACKETS:
            received[seq] = payload

    tx.close()
    rx.close()

    # ----- Score the run -----
    total_payload_bits = NUM_PACKETS * PAYLOAD_LEN * 8
    total_bit_errors = 0
    frame_errors = 0
    frames_missing = 0

    for seq in range(NUM_PACKETS):
        expected = payload_for(seq)
        got = received.get(seq)
        if got is None:
            frames_missing += 1
            frame_errors += 1
            total_bit_errors += PAYLOAD_LEN * 8
            continue
        be = popcount_xor(expected, got)
        if be > 0:
            frame_errors += 1
        total_bit_errors += be

    fer = frame_errors / NUM_PACKETS
    ber = total_bit_errors / total_payload_bits

    print(f"[{label}] sent={NUM_PACKETS}  recv={len(received)}  "
          f"missing={frames_missing}  bad={frame_errors - frames_missing}")
    print(f"[{label}] FER={fer:.6f}  BER={ber:.6f}")

    return {
        "label": label,
        "sent": NUM_PACKETS,
        "received": len(received),
        "missing": frames_missing,
        "frame_errors": frame_errors,
        "fer": fer,
        "ber": ber,
    }


# ---------------------------------------------------------------------------
# Plot + main
# ---------------------------------------------------------------------------

def plot_results(results):
    x = list(range(len(results)))
    labels = [r["label"] for r in results]

    floor_ber = 0.5 / (NUM_PACKETS * PAYLOAD_LEN * 8)
    floor_fer = 0.5 / NUM_PACKETS
    fers = [max(r["fer"], floor_fer) for r in results]
    bers = [max(r["ber"], floor_ber) for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, fers, marker="o", label="FER")
    plt.plot(x, bers, marker="s", label="BER")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.yscale("log")
    plt.ylabel("Error rate")
    plt.xlabel("Trial")
    plt.title("BER / FER vs power setting (Pluto -> RTL-SDR, AX.25)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved plot to {PLOT_PATH}")
    plt.show()


def main():
    print("=" * 64)
    print("BER / FER test harness  (Pluto TX -> RTL-SDR RX, AX.25)")
    print("=" * 64)
    print(f"Frames per trial : {NUM_PACKETS}")
    print(f"Payload size     : {PAYLOAD_LEN} B")
    print(f"AX.25 frame size : {len(build_pdu(0))} B "
          f"(16 hdr + 4 seq + {PAYLOAD_LEN} payload)")
    print(f"TX target        : {TX_HOST}:{TX_PORT}     (UDP_SERVER on plutosdr_tx)")
    print(f"RX listener      : {RX_BIND_HOST}:{RX_BIND_PORT}     (UDP_CLIENT from rtlsdr_rx)")
    print(f"Callsigns        : {SRC_CALL.strip()}-{SRC_SSID} -> "
          f"{DEST_CALL.strip()}-{DEST_SSID}")
    print()
    print("Make sure BOTH GNU Radio flowgraphs are RUNNING before each trial.")

    results = []
    for label in TRIAL_LABELS:
        input(f"\n>>> Set hardware to '{label}', confirm both flowgraphs "
              f"are running, then press Enter to begin the trial...")
        results.append(run_trial(label))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "sent", "received", "missing",
                        "frame_errors", "fer", "ber"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote results to {CSV_PATH}")

    plot_results(results)


if __name__ == "__main__":
    main()
