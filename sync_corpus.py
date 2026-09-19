#!/usr/bin/env python3
"""Align a record-corpus session using the sync pulse only.

Files are grouped by <corpus>_<index>_<mic>.wav (as written by
`audio.py record-corpus --sync-pulse`). For every index, each mic's file is
searched for the 1 kHz pulse:

  1. coarse:  sliding pulse-length average of (narrowband 1 kHz energy) /
              (neighbouring-band energy) over the whole file. The pulse is the
              window where that ratio peaks; its height over the file median is
              the detection score.
  2. refine:  within +/-80 ms of the coarse hit (the coarse STFT stage is
              ambiguous by about half a frame), fit a pulse-length box to the
              log ratio of the 1 kHz band envelope to its neighbouring bands.
              The ratio cancels broadband clicks and handling noise; fitting
              the whole box (mean inside minus mean just outside) lets both
              edges vote, which matters on mics whose tone envelope is noisy or
              rises slowly (e.g. a contact mic). That difference, in dB, is the
              box contrast.

A pulse is trusted only if both the detection score and the box contrast clear
their thresholds (--min-score, --min-contrast); otherwise that mic is reported
as unsynced for that prompt. Alignment uses the pulse alone, never the speech,
so it holds up in noisy environments: each mic's file is trimmed relative to
its own detected pulse, starting --lead seconds after the pulse ends, and all
mics of a prompt are cut to their common length, so sample n is the same
instant in every output file. Precision is limited by how sharply each mic
captures the pulse edges (a few ms on a contact mic). The corpus manifests
(<corpus>_prompts.tsv) are copied alongside, and sync_report.tsv lists every
pulse time, score, contrast and offset against --ref.

Usage:
    python sync_corpus.py 2026-09-18-lever-coffee-outside -o synced --ref mbp
    python sync_corpus.py DIR -o synced --mics contact,mbp
"""
import argparse
import collections
import glob
import os
import re
import shutil
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft

FILE_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d{4,})_(?P<mic>[^_]+(?:_ch\d+)?)\.wav$")


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: expected 16-bit PCM")
        rate, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return x, rate


def write_wav(path, x, rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.ascontiguousarray(x, dtype=np.int16).tobytes())


def coarse_pulse(x, rate, f0, dur):
    """Return (onset_s, score_db): where a dur-long f0 tone most stands out."""
    nper, hop = 1024, 64
    f, t, Z = stft(x.astype(np.float64), rate, nperseg=nper, noverlap=nper - hop, boundary=None)
    P = np.abs(Z) ** 2
    on = P[np.abs(f - f0) <= 16].sum(0)
    side_band = (np.abs(f - f0) >= 60) & (np.abs(f - f0) <= 250)
    side = P[side_band].mean(0) * max(1, np.count_nonzero(np.abs(f - f0) <= 16))
    ratio = 10 * np.log10((on + 1e-9) / (side + 1e-9))
    L = max(1, int(dur * rate / hop))
    if len(ratio) < L:
        return None, 0.0
    k = np.convolve(ratio, np.ones(L) / L, "valid")
    j = int(k.argmax())
    return t[j] - nper / 2 / rate, float(k[j] - np.median(k))


def refine_onset(x, rate, coarse_s, f0, dur, search=0.08, guard=0.05):
    """Return (onset_s, contrast_db) from a dur-long box fit to the f0 band envelope.

    contrast_db is how much more the tone band dominates its neighbours inside
    the box than in the `guard` seconds either side.
    """
    a = max(0, int((coarse_s - search - guard - 0.05) * rate))
    b = min(len(x), int((coarse_s + dur + search + guard + 0.05) * rate))
    seg = x[a:b].astype(np.float64)
    win = np.ones(max(1, int(0.01 * rate))) / max(1, int(0.01 * rate))

    def band_env(lo, hi):
        sos = butter(4, [lo, hi], "bandpass", fs=rate, output="sos")
        return np.convolve(np.abs(sosfiltfilt(sos, seg)), win, "same")

    # Log ratio of the tone band to its neighbours: broadband clicks (handling
    # noise, taps) raise both and cancel, while the steady tone stands out.
    on = band_env(f0 - 50, f0 + 50)
    side = 0.5 * (band_env(f0 - 350, f0 - 150) + band_env(f0 + 150, f0 + 350))
    env = np.log10((on + 1e-3) / (side + 1e-3))
    c = np.concatenate(([0.0], np.cumsum(env)))
    D, G = int(dur * rate), int(guard * rate)
    best, best_score = None, -np.inf  # score: mean log-ratio inside minus outside
    for s in range(int((coarse_s - search) * rate) - a, int((coarse_s + search) * rate) - a + 1):
        if s - G < 0 or s + D + G > len(env):
            continue
        inside = (c[s + D] - c[s]) / D
        outside = (c[s] - c[s - G] + c[s + D + G] - c[s + D]) / (2 * G)
        if inside - outside > best_score:
            best, best_score = s, inside - outside
    if best is None:
        return coarse_s, 0.0
    return (a + best) / rate, float(20 * best_score)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("indir", help="record-corpus output directory")
    p.add_argument("-o", "--out", required=True, help="directory for aligned WAVs")
    p.add_argument("--ref", default="mbp",
                   help="mic that offsets are reported against in the log and report "
                        "(alignment itself always uses each mic's own pulse)")
    p.add_argument("--mics", help="comma-separated mics to include (default: all). Outputs are "
                                  "cut to the common length, so leave out a mic whose pulse "
                                  "lands late in its file")
    p.add_argument("--tone", type=float, default=1000.0, help="pulse frequency (Hz)")
    p.add_argument("--pulse-duration", type=float, default=0.3, help="pulse length (s)")
    p.add_argument("--min-score", type=float, default=9.0,
                   help="minimum detection score (dB over file median) to trust a pulse")
    p.add_argument("--lead", type=float, default=0.15,
                   help="seconds after the pulse ends where aligned output starts")
    p.add_argument("--min-contrast", type=float, default=6.0,
                   help="minimum box contrast (dB, envelope inside vs just outside the pulse)")
    args = p.parse_args()

    wanted = set(args.mics.split(",")) if args.mics else None
    groups = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.indir, "*.wav"))):
        m = FILE_RE.match(os.path.basename(path))
        if m and (not wanted or m["mic"] in wanted):
            groups[(m["prefix"], m["idx"])][m["mic"]] = path
    if not groups:
        sys.exit(f"no <corpus>_<index>_<mic>.wav files in {args.indir}")
    os.makedirs(args.out, exist_ok=True)
    for manifest in glob.glob(os.path.join(args.indir, "*_prompts.tsv")):
        shutil.copy2(manifest, args.out)

    report = open(os.path.join(args.out, "sync_report.tsv"), "w")
    report.write("prefix\tindex\tmic\tpulse_s\tscore_db\tcontrast_db\toffset_vs_ref_ms\tstatus\n")
    mics = sorted({mic for g in groups.values() for mic in g})
    print(f"{len(groups)} prompts, mics: {', '.join(mics)}; reference: {args.ref}")
    w = max(len(f"{prefix}_{idx}") for prefix, idx in groups)
    print(f"{'prompt':<{w}} " + " ".join(f"{m:>28}" for m in mics))

    counts = collections.Counter()
    for (prefix, idx), files in sorted(groups.items()):
        found, audio = {}, {}
        for mic, path in files.items():
            x, rate = read_wav(path)
            audio[mic] = (x, rate)
            onset, score = coarse_pulse(x, rate, args.tone, args.pulse_duration)
            t = contrast = None
            if onset is not None and score >= args.min_score:
                t, contrast = refine_onset(x, rate, onset, args.tone, args.pulse_duration)
                if contrast < args.min_contrast:
                    t = None
            found[mic] = (t, score, contrast)

        ok = {mic for mic, (t, _, _) in found.items() if t is not None}
        rates = {audio[m][1] for m in ok}
        if len(rates) > 1:
            sys.exit(f"{prefix}_{idx}: mixed sample rates {rates}; resample first")
        ref_t = found.get(args.ref, (None,))[0]

        # Start of each synced segment, in that file's own samples.
        starts = {m: int(round((found[m][0] + args.pulse_duration + args.lead) * audio[m][1]))
                  for m in ok}
        n_common = min((len(audio[m][0]) - starts[m] for m in ok), default=0)
        cells = []
        for mic in mics:
            if mic not in files:
                cells.append(f"{'-':>28}")
                continue
            t, score, contrast = found[mic]
            if t is None:
                status = "no-pulse"
            elif n_common <= 0:
                status = "too-short"
            else:
                status = "ok"
                x, rate = audio[mic]
                write_wav(os.path.join(args.out, f"{prefix}_{idx}_{mic}.wav"),
                          x[starts[mic]:starts[mic] + n_common], rate)
            counts[(mic, status)] += 1
            off = "" if t is None or ref_t is None else f"{1000 * (t - ref_t):+.1f}"
            con = "" if contrast is None else f"{contrast:.1f}"
            report.write(f"{prefix}\t{idx}\t{mic}\t{'' if t is None else f'{t:.4f}'}\t"
                         f"{score:.1f}\t{con}\t{off}\t{status}\n")
            label = (f"{t:6.3f}s {score:4.1f}/{contrast:4.1f}dB {off:>7}" if t is not None
                     else f"NO PULSE ({score:4.1f}/{con or '-':>4}dB)")
            cells.append(f"{label:>28}")
        print(f"{prefix + '_' + idx:<{w}} " + " ".join(cells))

    report.close()
    print("-" * 60)
    for mic in mics:
        print(f"{mic:>10}: {counts[(mic, 'ok')]} synced, "
              f"{counts[(mic, 'no-pulse')]} no pulse, {counts[(mic, 'too-short')]} too short")
    print(f"Wrote aligned WAVs + sync_report.tsv to {args.out}")


if __name__ == "__main__":
    main()
