"""
FER vs. static antenna pointing-angle experiment.

Experiment #2 for the PULSE-A paper: instead of sweeping TX power,
hold TX attenuation constant and manually rotate one antenna to a
series of angles. This simulates a CubeSat pass where the ground
station beam is off-boresight.

Usage
-----
1. Start the RX flowgraph:
       python rtlsdr_rx.py

2. Start the TX flowgraph at the chosen attenuation:
       python plutosdr_tx.py --attn <TX_ATTN_DB>

3. Run this script:
       python fer_vs_pointing.py

   The script will prompt you before each trial to rotate the antenna
   to the specified angle, then wait for you to press Enter.

Outputs
-------
  pointing_results.csv   one row per angle trial
  pointing_fer_plot.png  FER vs pointing angle (bar + line chart)

Configuration
-------------
Set TX_ATTN_DB to an attenuation that gives FER ≈ 0 at boresight
(0° / antennas co-aligned). From the Task 1 BER sweep, pick a value
well below the breakdown region so you have margin to observe
degradation as angle increases. A reasonable starting guess is 20 dB.

ANGLE_DEGREES lists the angles at which trials are run. The human
rotates the RX antenna (or TX, but pick one and be consistent).
Keep the pivot antenna vertical at 0° and record absolute deflection
from the co-linear alignment position.
"""

import csv
import random
import socket
import struct
import time

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TX_HOST = "127.0.0.1"
TX_PORT = 52001

RX_BIND_HOST = "127.0.0.1"
RX_BIND_PORT = 52002

# TX attenuation for ALL trials — set below the FER cliff from Task 1.
# This should give FER ≈ 0 at 0° so you can see it degrade with angle.
# Update this after running the Task 1 sweep.
TX_ATTN_DB = 40.0  # from Task 1: FER=0% at 40 dB — clean baseline, ~5 dB below cliff (~45 dB)

# Antenna pointing angles to sweep (degrees from boresight/co-linear).
ANGLE_DEGREES = [0, 10, 20, 30, 45, 60, 90]

NUM_PACKETS = 500
PAYLOAD_LEN = 60

INTER_PACKET_DELAY = 0.05   # seconds
RX_DRAIN_TIME = 8.0         # seconds to drain stragglers after last TX
RX_RECV_TIMEOUT = 0.2

DEST_CALL = "DEST  "
DEST_SSID = 0
SRC_CALL  = "SRC   "
SRC_SSID  = 0

CSV_PATH  = "pointing_results.csv"
PLOT_PATH = "pointing_fer_plot.png"


# ---------------------------------------------------------------------------
# AX.25 frame helpers (shared with ber_fer_test.py)
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

def run_trial(angle_deg: float) -> dict:
    label = f"{angle_deg}°"

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
        "angle_deg": angle_deg,
        "label": label,
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
    angles = [r["angle_deg"] for r in results]
    fers   = [r["fer"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = max(4, min(8, (angles[-1] - angles[0]) / len(angles) * 0.6))
    ax.bar(angles, fers, width=bar_width, alpha=0.4, color="steelblue",
           label="FER (bar)")
    ax.plot(angles, fers, marker="o", color="steelblue", label="FER (line)")

    ax.set_xlabel("Antenna mis-pointing angle (degrees from boresight)")
    ax.set_ylabel("Frame Error Rate")
    ax.set_title(
        f"FER vs. Antenna Pointing Angle\n"
        f"(Pluto → RTL-SDR, AX.25 GFSK 9600 baud, TX attn={TX_ATTN_DB} dB)"
    )
    ax.set_xticks(angles)
    ax.set_xlim(-5, max(angles) + 10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved plot to {PLOT_PATH}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("FER vs. Antenna Pointing Angle")
    print("(Pluto TX → RTL-SDR RX, AX.25 GFSK 9600 baud)")
    print("=" * 64)
    print(f"TX attenuation  : {TX_ATTN_DB} dB  (held constant)")
    print(f"Angles to sweep : {ANGLE_DEGREES} degrees")
    print(f"Frames per trial: {NUM_PACKETS}")
    print(f"Payload size    : {PAYLOAD_LEN} B")
    print()
    print("SETUP:")
    print(f"  1. Start RX  : python -u rtlsdr_rx.py > rx.log 2>&1 &")
    print(f"  2. Start TX  : python -u plutosdr_tx.py --attn {TX_ATTN_DB} > tx.log 2>&1 &")
    print()
    print("Before each trial you will be prompted to rotate the antenna.")
    print("Keep the same pivot axis throughout. Measure from the co-aligned")
    print("(0°) position — both antennas pointing straight up / co-linear.")
    input("\nPress Enter when both flowgraphs are running ...")

    results = []
    for angle in ANGLE_DEGREES:
        print()
        print(f"--- Next trial: {angle}° ---")
        if angle == 0:
            input(f"  Align antennas to 0° (co-linear / both upright). Press Enter...")
        else:
            input(f"  Rotate the antenna to {angle}° from boresight. Press Enter...")

        results.append(run_trial(angle))

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["angle_deg", "label", "sent", "received", "missing",
                        "frame_errors", "fer", "ber"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote results to {CSV_PATH}")

    plot_results(results)

    # Repeatability check: re-run 0° at the end
    redo = input("\nRe-run 0° trial for repeatability check? [y/N] ").strip().lower()
    if redo == "y":
        input("  Realign antennas to 0°. Press Enter...")
        r = run_trial(0)
        original_0 = next((x for x in results if x["angle_deg"] == 0), None)
        if original_0:
            drift = abs(r["fer"] - original_0["fer"])
            print(f"\nRepeatability: original FER={original_0['fer']:.4f}, "
                  f"repeat FER={r['fer']:.4f}, delta={drift:.4f}")
            if drift > 0.05:
                print("  WARNING: drift > 5 pp — check antenna/RX gain stability.")
            else:
                print("  OK.")


if __name__ == "__main__":
    main()
