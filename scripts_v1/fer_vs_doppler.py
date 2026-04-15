"""
FER vs. TX frequency offset (artificial Doppler shift) experiment.

Experiment #3 for the PULSE-A paper: hold TX power constant and sweep
the Pluto's TX center frequency ±10 kHz around the nominal 433.350 MHz.
This simulates the Doppler shift a LEO CubeSat produces during a pass
(worst case ≈ ±10 kHz at 433 MHz for a 7 km/s spacecraft).

The RX is NOT retuned — the whole point is to measure how well the
receiver's timing/frequency tracking copes with the offset.

Usage
-----
1. Start the RX flowgraph:
       python rtlsdr_rx.py

2. Start the TX flowgraph at the chosen attenuation:
       python plutosdr_tx.py --attn <TX_ATTN_DB>

3. Run this script (it shifts TX frequency programmatically):
       python fer_vs_doppler.py

Outputs
-------
  doppler_results.csv   one row per frequency offset
  doppler_fer_plot.png  FER vs. frequency offset (symmetric about 0)

Configuration
-------------
TX_ATTN_DB — attenuation where Task 1 showed FER ≈ 0 at 0 Hz offset.
FREQ_OFFSETS_HZ — list of signed Hz offsets from NOMINAL_FREQ_HZ.
  Positive = TX above nominal (approaching spacecraft)
  Negative = TX below nominal (receding spacecraft)
  Recommended range: ±10 kHz in 1–2 kHz steps.
SETTLE_TIME — seconds to wait after changing TX frequency for the
  AD9361 LO to stabilise and the RX tracking loop to re-acquire.
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

# Center (nominal) TX frequency in Hz
NOMINAL_FREQ_HZ = 433_350_000

# TX attenuation held constant for all trials (set from Task 1 results).
TX_ATTN_DB = 40.0  # from Task 1: FER=0% at 40 dB — clean baseline, ~5 dB below cliff (~45 dB)

# Signed frequency offsets from nominal (Hz).
# Symmetric sweep ±10 kHz; include 0 for the baseline.
FREQ_OFFSETS_HZ = [
    -10_000,
    -8_000,
    -6_000,
    -4_000,
    -2_000,
    0,
    2_000,
    4_000,
    6_000,
    8_000,
    10_000,
]

NUM_PACKETS = 500
PAYLOAD_LEN = 60

INTER_PACKET_DELAY = 0.05   # seconds
RX_DRAIN_TIME = 8.0         # seconds to drain stragglers
RX_RECV_TIMEOUT = 0.2

# Time to wait after changing TX frequency.
# The AD9361 LO synthesizer settles in <1 ms but the RX PFB clock sync
# and any software-defined AFC need time to re-acquire. 2 s is conservative.
SETTLE_TIME = 2.0

DEST_CALL = "DEST  "
DEST_SSID = 0
SRC_CALL  = "SRC   "
SRC_SSID  = 0

CSV_PATH  = "doppler_results.csv"
PLOT_PATH = "doppler_fer_plot.png"


# ---------------------------------------------------------------------------
# Pluto TX frequency control
# ---------------------------------------------------------------------------

def set_pluto_tx_freq(hz: int):
    """Set the Pluto TX LO frequency via iio_attr.

    The AD9361 TX LO is 'altvoltage1' in the IIO device tree.
    Value is in Hz (integer).
    """
    subprocess.run(
        ["iio_attr", "-u", PLUTO_URI, "-c", "ad9361-phy",
         "altvoltage1", "frequency", str(int(hz))],
        check=True,
        capture_output=True,
    )


def get_pluto_tx_freq() -> int:
    """Read back the current TX LO frequency for verification."""
    result = subprocess.run(
        ["iio_attr", "-u", PLUTO_URI, "-c", "ad9361-phy",
         "altvoltage1", "frequency"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().splitlines():
        try:
            return int(line.split()[0])
        except (ValueError, IndexError):
            continue
    raise RuntimeError(f"Could not parse TX frequency: {result.stdout!r}")


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

def run_trial(offset_hz: int) -> dict:
    label = f"{offset_hz:+d} Hz"

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
        "offset_hz": offset_hz,
        "tx_freq_hz": NOMINAL_FREQ_HZ + offset_hz,
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
    offsets_khz = [r["offset_hz"] / 1000 for r in results]
    fers = [r["fer"] for r in results]

    floor_fer = 0.5 / NUM_PACKETS
    fers_floored = [max(f, floor_fer) for f in fers]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(offsets_khz, fers_floored, marker="o", color="darkorange",
            label="FER")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="Nominal freq")

    ax.set_yscale("log")
    ax.set_xlabel("TX frequency offset (kHz)")
    ax.set_ylabel("Frame Error Rate")
    ax.set_title(
        f"FER vs. TX Frequency Offset (Artificial Doppler)\n"
        f"(Pluto → RTL-SDR, AX.25 GFSK 9600 baud, TX attn={TX_ATTN_DB} dB, "
        f"nominal {NOMINAL_FREQ_HZ/1e6:.3f} MHz)"
    )
    ax.grid(True, which="both", linestyle=":")
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
    print("FER vs. TX Frequency Offset  (artificial Doppler sweep)")
    print("(Pluto TX → RTL-SDR RX, AX.25 GFSK 9600 baud)")
    print("=" * 64)
    print(f"Nominal TX freq : {NOMINAL_FREQ_HZ/1e6:.3f} MHz")
    print(f"TX attenuation  : {TX_ATTN_DB} dB  (held constant)")
    print(f"Offsets to sweep: {FREQ_OFFSETS_HZ} Hz")
    print(f"Frames per trial: {NUM_PACKETS}")
    print(f"Payload size    : {PAYLOAD_LEN} B")
    print(f"Settle time     : {SETTLE_TIME} s per step")
    print()
    print("Make sure BOTH GNU Radio flowgraphs are already running:")
    print(f"  python -u rtlsdr_rx.py > rx.log 2>&1 &")
    print(f"  python -u plutosdr_tx.py --attn {TX_ATTN_DB} > tx.log 2>&1 &")
    input("\nPress Enter when both flowgraphs are running ...")

    # Restore nominal frequency when we finish or if interrupted
    import atexit
    import signal as _signal

    def restore_freq():
        try:
            print(f"\nRestoring TX to nominal {NOMINAL_FREQ_HZ/1e6:.3f} MHz ...")
            set_pluto_tx_freq(NOMINAL_FREQ_HZ)
        except Exception:
            pass

    atexit.register(restore_freq)
    _signal.signal(_signal.SIGINT, lambda *_: (restore_freq(), exit(0)))
    _signal.signal(_signal.SIGTERM, lambda *_: (restore_freq(), exit(0)))

    results = []
    for offset in FREQ_OFFSETS_HZ:
        tx_freq = NOMINAL_FREQ_HZ + offset
        print(f"\n--- Setting TX to {tx_freq/1e6:.4f} MHz "
              f"(offset {offset:+d} Hz) ---")
        set_pluto_tx_freq(tx_freq)
        time.sleep(SETTLE_TIME)

        # Verify
        try:
            actual = get_pluto_tx_freq()
            print(f"    Readback: {actual/1e6:.4f} MHz "
                  f"(delta = {actual - tx_freq:+d} Hz)")
        except Exception as e:
            print(f"    [WARN] Could not read back frequency: {e}")

        results.append(run_trial(offset))

    # Restore nominal before writing output
    restore_freq()

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["offset_hz", "tx_freq_hz", "label", "sent", "received",
                        "missing", "frame_errors", "fer", "ber"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote results to {CSV_PATH}")

    plot_results(results)

    # Print summary — find the ±offset where FER first exceeds 5%
    print("\n--- Doppler tolerance summary ---")
    baseline = next((r for r in results if r["offset_hz"] == 0), None)
    if baseline:
        print(f"Baseline (0 Hz): FER={baseline['fer']:.4f}")
    threshold = 0.05
    tolerant = [r for r in results if r["fer"] <= threshold]
    if tolerant:
        max_offset = max(abs(r["offset_hz"]) for r in tolerant)
        print(f"FER ≤ {threshold:.0%} for offsets within ±{max_offset} Hz "
              f"(±{max_offset/1000:.1f} kHz)")
    else:
        print(f"FER exceeded {threshold:.0%} at all tested offsets — "
              f"check gain / rx-offset settings.")


if __name__ == "__main__":
    main()
