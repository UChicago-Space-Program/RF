"""
Automated BER/FER sweep for the PlutoSDR → RTL-SDR + AX.25 loopback.

Identical measurement methodology to ber_fer_test.py, but:
  - Takes a list of attenuation values (floats) rather than prompting
  - Sets the Pluto's TX attenuation programmatically via iio_attr between
    trials, so the TX flowgraph runs continuously and re-init time is zero
  - Runs all trials back-to-back without human intervention

Usage
-----
1. Start the RX flowgraph in one terminal:
       python rtlsdr_rx.py --gain 49.6

2. Start the TX flowgraph in another terminal:
       python plutosdr_tx.py --attn 0

3. Run this script (it controls the Pluto attenuation itself):
       python ber_fer_test_auto.py

Outputs: ber_results.csv, ber_fer_plot.png  (same format as ber_fer_test.py)
"""

import csv
import random
import socket
import struct
import subprocess
import time

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TX_HOST = "127.0.0.1"
TX_PORT = 52001

RX_BIND_HOST = "127.0.0.1"
RX_BIND_PORT = 52002

PLUTO_URI = "ip:192.168.2.1"

# Attenuation sweep (dB). Adjust after a probe run to bracket the breakeven.
# These defaults are a wide probe sweep; tighten around the cliff after first run.
ATTN_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80]

NUM_PACKETS = 500           # Use 1000 for publication quality once range is known
PAYLOAD_LEN = 60

INTER_PACKET_DELAY = 0.05   # seconds; 9600 baud needs breathing room
RX_DRAIN_TIME = 8.0         # seconds to drain stragglers after last TX
RX_RECV_TIMEOUT = 0.2

# Wait after changing attenuation to let the RF chain settle
ATTN_SETTLE_TIME = 1.0

DEST_CALL = "DEST  "
DEST_SSID = 0
SRC_CALL  = "SRC   "
SRC_SSID  = 0

CSV_PATH  = "ber_results.csv"
PLOT_PATH = "ber_fer_plot.png"


# ---------------------------------------------------------------------------
# Pluto attenuation control
# ---------------------------------------------------------------------------

def set_pluto_attn(db: float):
    """Set the Pluto TX attenuation via iio_attr (hardwaregain is negative)."""
    subprocess.run(
        ["iio_attr", "-u", PLUTO_URI, "-c", "ad9361-phy",
         "-o", "voltage0", "hardwaregain", str(-abs(db))],
        check=True,
        capture_output=True,
    )


def get_pluto_attn() -> float:
    """Read back the current TX attenuation for verification."""
    result = subprocess.run(
        ["iio_attr", "-u", PLUTO_URI, "-c", "ad9361-phy",
         "-o", "voltage0", "hardwaregain"],
        check=True,
        capture_output=True,
        text=True,
    )
    # Output is like "-30.000000 dB\n"
    for line in result.stdout.strip().splitlines():
        try:
            return -float(line.split()[0])   # convert hardwaregain to attenuation
        except (ValueError, IndexError):
            continue
    raise RuntimeError(f"Could not parse hardwaregain: {result.stdout!r}")


# ---------------------------------------------------------------------------
# AX.25 header + frame helpers  (identical to ber_fer_test.py)
# ---------------------------------------------------------------------------

def encode_ax25_address(callsign: str, ssid: int, last: bool) -> bytes:
    cs = (callsign.upper() + "      ")[:6]
    out = bytearray(c << 1 for c in cs.encode("ascii"))
    out.append(0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0))
    return bytes(out)


def build_ax25_header() -> bytes:
    dest = encode_ax25_address(DEST_CALL, DEST_SSID, last=False)
    src  = encode_ax25_address(SRC_CALL,  SRC_SSID,  last=True)
    return dest + src + bytes([0x03, 0xF0])


AX25_HEADER = build_ax25_header()
HEADER_LEN  = len(AX25_HEADER)

_POPCOUNT = bytes(bin(i).count("1") for i in range(256))


def popcount_xor(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    diff = sum(_POPCOUNT[a[i] ^ b[i]] for i in range(n))
    diff += abs(len(a) - len(b)) * 8
    return diff


def payload_for(seq: int) -> bytes:
    rng = random.Random(seq)
    return bytes(rng.getrandbits(8) for _ in range(PAYLOAD_LEN))


def build_pdu(seq: int) -> bytes:
    return AX25_HEADER + struct.pack(">I", seq) + payload_for(seq)


def parse_received(data: bytes):
    if len(data) < HEADER_LEN + 4:
        return None, None
    body = data[HEADER_LEN:]
    seq = struct.unpack(">I", body[:4])[0]
    payload = body[4: 4 + PAYLOAD_LEN]
    return seq, payload


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_trial(label: str, attn_db: float) -> dict:
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((RX_BIND_HOST, RX_BIND_PORT))
    rx.settimeout(RX_RECV_TIMEOUT)

    received = {}

    print(f"\n[{label}] Sending {NUM_PACKETS} PDUs ...")
    for seq in range(NUM_PACKETS):
        tx.sendto(build_pdu(seq), (TX_HOST, TX_PORT))
        if INTER_PACKET_DELAY:
            time.sleep(INTER_PACKET_DELAY)

    print(f"[{label}] Draining RX for {RX_DRAIN_TIME:.1f}s ...")
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

    total_bits = NUM_PACKETS * PAYLOAD_LEN * 8
    total_errors = 0
    frame_errors = 0
    frames_missing = 0

    for seq in range(NUM_PACKETS):
        expected = payload_for(seq)
        got = received.get(seq)
        if got is None:
            frames_missing += 1
            frame_errors += 1
            total_errors += PAYLOAD_LEN * 8
        else:
            be = popcount_xor(expected, got)
            if be > 0:
                frame_errors += 1
            total_errors += be

    fer = frame_errors / NUM_PACKETS
    ber = total_errors / total_bits

    print(f"[{label}] sent={NUM_PACKETS}  recv={len(received)}  "
          f"missing={frames_missing}  bad={frame_errors - frames_missing}")
    print(f"[{label}] FER={fer:.6f}  BER={ber:.6f}")

    return {
        "label": label,
        "attn_db": attn_db,
        "sent": NUM_PACKETS,
        "received": len(received),
        "missing": frames_missing,
        "frame_errors": frame_errors,
        "fer": fer,
        "ber": ber,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results):
    labels = [r["label"] for r in results]
    x = list(range(len(results)))

    floor_ber = 0.5 / (NUM_PACKETS * PAYLOAD_LEN * 8)
    floor_fer = 0.5 / NUM_PACKETS
    fers = [max(r["fer"], floor_fer) for r in results]
    bers = [max(r["ber"], floor_ber) for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(x, fers, marker="o", label="FER")
    plt.plot(x, bers, marker="s", label="BER")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.yscale("log")
    plt.ylabel("Error rate")
    plt.xlabel("TX attenuation")
    plt.title("BER / FER vs TX attenuation (Pluto → RTL-SDR, AX.25 GFSK 9600 baud)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved plot to {PLOT_PATH}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Automated BER/FER sweep  (Pluto TX → RTL-SDR RX, AX.25)")
    print("=" * 64)
    print(f"Attenuation sweep : {ATTN_VALUES} dB")
    print(f"Frames per trial  : {NUM_PACKETS}")
    print(f"Payload size      : {PAYLOAD_LEN} B")
    print(f"Pluto URI         : {PLUTO_URI}")
    print()
    print("Make sure BOTH GNU Radio flowgraphs are already running:")
    print("  python -u rtlsdr_rx.py  > rx.log 2>&1 &")
    print("  python -u plutosdr_tx.py --attn 0  > tx.log 2>&1 &")
    input("\nPress Enter when both flowgraphs are running ...")

    results = []
    for attn in ATTN_VALUES:
        label = f"{attn} dB attn"
        print(f"\n--- Setting Pluto TX attenuation to {attn} dB ---")
        set_pluto_attn(attn)
        time.sleep(ATTN_SETTLE_TIME)

        # Verify the attenuation was accepted
        try:
            actual = get_pluto_attn()
            print(f"    Readback: {actual:.2f} dB (set {attn} dB)")
        except Exception as e:
            print(f"    [WARN] Could not read back attenuation: {e}")

        results.append(run_trial(label, attn))

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "attn_db", "sent", "received", "missing",
                        "frame_errors", "fer", "ber"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote results to {CSV_PATH}")

    plot_results(results)


if __name__ == "__main__":
    main()
