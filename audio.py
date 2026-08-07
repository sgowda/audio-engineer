#!/usr/bin/env python3
"""audio_engineer.py — a growing toolbox for all things audio.

Subcommands:
    plot        Plot the waveform of one or more WAV files.
    plot-same   Plot all waveforms overlaid on a single subplot.
    spectrogram Plot the spectrogram of one or more WAV files.
    gain        Apply a digital gain (dB) to one or more WAV files.
    gain-equalize  Match each file's level to a reference (90th percentile).
    denoise     Suppress environmental noise with DeepFilterNet3.
    record      Record from one or more microphones to an output dir.
    loopback    Route a live input device straight to an output device.
"""
import argparse
import glob
import os
import re
import threading
import wave
try:
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import medfilt

# Recording parameters mirror record_multi_audio.py.
RECORD_CHUNK_SIZE = 1024
RECORD_SAMPLE_WIDTH = 2  # bytes, paInt16
DEFAULT_RECORD_RATE = 16000


def load_wav(path):
    """Read a WAV file, returning (rate, data, channels) with data as float."""
    rate, data = wav.read(path)
    channels = 1 if data.ndim == 1 else data.shape[1]
    return rate, data, channels


def expand_paths(patterns):
    """Expand each pattern as a glob, preserving order and dropping duplicates."""
    paths = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"Warning: no files matched '{pattern}'")
        for m in matches:
            if m not in seen:
                seen.add(m)
                paths.append(m)
    return paths


def plot_wav_file(wavfile):
    """Plot every channel of a single WAV file as a stack of subplots."""
    rate, data, channels = load_wav(wavfile)

    # Time axis in seconds
    t = np.arange(data.shape[0]) / rate

    plt.figure(figsize=(12, 4 * channels))
    for ch in range(channels):
        samples = data if channels == 1 else data[:, ch]
        ax = plt.subplot(channels, 1, ch + 1)
        ax.plot(t, samples, linewidth=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{wavfile} — channel {ch} ({rate} Hz)")
        ax.grid(True, alpha=0.3)
    plt.xlabel("Time [s]")
    plt.tight_layout()


def plot_wav_files_overlaid(wavfiles):
    """Plot every channel of every WAV file overlaid on a single subplot."""
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 1, 1)
    for wavfile in wavfiles:
        rate, data, channels = load_wav(wavfile)
        t = np.arange(data.shape[0]) / rate
        for ch in range(channels):
            samples = data if channels == 1 else data[:, ch]
            label = wavfile if channels == 1 else f"{wavfile} — ch {ch}"
            ax.plot(t, samples, linewidth=0.5, alpha=0.7, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveforms")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    plt.tight_layout()


def spectrogram_wav_file(wavfile):
    """Plot every channel of a single WAV file as a stack of spectrogram subplots."""
    rate, data, channels = load_wav(wavfile)

    plt.figure(figsize=(12, 4 * channels))
    for ch in range(channels):
        samples = data if channels == 1 else data[:, ch]
        ax = plt.subplot(channels, 1, ch + 1)
        Pxx, freqs, bins, im = ax.specgram(
            samples,
            NFFT=1024,       # window size
            Fs=rate,         # sampling rate
            noverlap=512,    # overlap between windows
            cmap="viridis",
        )
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"{wavfile} — channel {ch} ({rate} Hz)")
        plt.colorbar(im, ax=ax).set_label("Intensity [dB]")
    plt.xlabel("Time [s]")
    plt.tight_layout()


def apply_gain_db(data, db):
    """Scale samples by a gain in dB, clipping back into the original dtype range."""
    dtype = data.dtype
    gain = 10 ** (db / 20.0)

    # Work in float to avoid intermediate overflow, then clip into range.
    scaled = data.astype(np.float64) * gain
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = np.clip(scaled, info.min, info.max)
    else:
        # Floating-point WAVs are conventionally in [-1.0, 1.0]
        scaled = np.clip(scaled, -1.0, 1.0)
    return scaled.astype(dtype)


def gain_output_path(wavfile, db):
    """Build an output path next to the original, with the gain appended to the stem."""
    base, ext = os.path.splitext(wavfile)
    # e.g. mic1.wav + 6 dB -> mic1_+6dB.wav ; -3.5 dB -> mic1_-3.5dB.wav
    return f"{base}_{db:+g}dB{ext}"


def cmd_gain(args):
    paths = expand_paths(args.wavfiles)
    if not paths:
        raise SystemExit("No WAV files to process.")

    for wavfile in paths:
        rate, data = wav.read(wavfile)
        out_data = apply_gain_db(data, args.db)
        out_path = gain_output_path(wavfile, args.db)
        wav.write(out_path, rate, out_data)
        print(f"Applied {args.db:+g} dB to {wavfile} -> {out_path}")


def percentile_level(data, pct):
    """Magnitude at the given percentile of |samples|, across all channels."""
    return float(np.percentile(np.abs(data.astype(np.float64)), pct))


def cmd_gain_equalize(args):
    ref_rate, ref_data = wav.read(args.reference)
    ref_level = percentile_level(ref_data, args.percentile)
    if ref_level <= 0:
        raise SystemExit(
            f"Reference {args.reference} has a {args.percentile:g}th-percentile level of 0; "
            "cannot equalize to it.")

    ref_abs = os.path.abspath(args.reference)
    paths = expand_paths(args.wavfiles)
    if not paths:
        raise SystemExit("No WAV files to equalize.")

    print(f"Reference {args.reference}: {args.percentile:g}th pct = {ref_level:.1f}")
    for wavfile in paths:
        if os.path.abspath(wavfile) == ref_abs:
            continue  # don't equalize the reference against itself
        rate, data = wav.read(wavfile)
        level = percentile_level(data, args.percentile)
        if level <= 0:
            print(f"Skipping {wavfile}: {args.percentile:g}th-percentile level is 0")
            continue
        db = 20.0 * np.log10(ref_level / level)
        out_data = apply_gain_db(data, db)
        base, ext = os.path.splitext(wavfile)
        out_path = f"{base}_eq{ext}"
        wav.write(out_path, rate, out_data)
        print(f"{wavfile}: {level:.1f} -> {ref_level:.1f} ({db:+.2f} dB) -> {out_path}")


def repair_overflow_glitches(audio, frac=0.5, kernel=5):
    """Repair integer-overflow wraparound spikes by interpolation.

    When a sample exceeds the codec's range it wraps to the opposite rail, so an
    otherwise-smooth waveform shows a one- or few-sample jump from near the max to
    near the min. We flag samples whose deviation from a local median exceeds `frac`
    of the full peak-to-peak swing and replace each flagged run by linear
    interpolation across the surrounding good samples.

    `audio` is float in [-1, 1], shape (channels, samples). Returns
    (repaired_audio, num_samples_fixed).
    """
    repaired = audio.copy()
    thresh = frac * 2.0  # full peak-to-peak span of a [-1, 1] signal
    total = 0
    for c in range(repaired.shape[0]):
        x = repaired[c]
        if x.size < 3:
            continue
        med = medfilt(x, kernel_size=kernel)
        bad = np.abs(x - med) > thresh
        if bad.sum() == 0 or (~bad).sum() < 2:
            continue
        idx = np.arange(x.size)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        total += int(bad.sum())
    return repaired, total


def _load_deepfilternet():
    """Import DeepFilterNet's model API, shimming an old torchaudio symbol it expects.

    DeepFilterNet 0.5.x imports `torchaudio.backend.common.AudioMetaData`, which newer
    torchaudio removed. We only need the model (init_df/enhance), so we inject a stub to
    let the import succeed and do audio I/O ourselves (see cmd_denoise).
    """
    import sys
    import types
    if "torchaudio.backend.common" not in sys.modules:
        shim = types.ModuleType("torchaudio.backend.common")
        shim.AudioMetaData = type("AudioMetaData", (), {})
        sys.modules["torchaudio.backend.common"] = shim
    from df.enhance import enhance, init_df
    return init_df, enhance


def cmd_denoise(args):
    try:
        import torch
        import torchaudio.functional as AF
        init_df, enhance = _load_deepfilternet()
    except ImportError as exc:
        raise SystemExit(
            f"The 'denoise' command requires DeepFilterNet and torch "
            f"(pip install -r requirements.txt). [{exc}]")

    paths = expand_paths(args.wavfiles)
    if not paths:
        raise SystemExit("No WAV files to denoise.")

    # DeepFilterNet3 is the default model shipped with the package; it runs at 48 kHz.
    model, df_state, _ = init_df()
    model_sr = df_state.sr()

    for wavfile in paths:
        in_sr, data = wav.read(wavfile)
        # Normalize to float32 in [-1, 1], shape (channels, samples).
        if np.issubdtype(data.dtype, np.integer):
            peak = float(np.iinfo(data.dtype).max + 1)
            audio = data.astype(np.float32) / peak
        else:
            audio = data.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T  # (samples, channels) -> (channels, samples)

        # Smooth out integer-overflow wraparound spikes before denoising.
        audio, n_fixed = repair_overflow_glitches(audio)
        if n_fixed:
            print(f"  repaired {n_fixed} overflow sample(s) in {wavfile}")

        # Optional pre-gain to lift quiet recordings above DeepFilterNet's noise
        # floor. Applied in-memory only; the gained signal is never written out.
        if args.gain:
            audio = np.clip(audio * (10.0 ** (args.gain / 20.0)), -1.0, 1.0)

        wave_t = torch.from_numpy(np.ascontiguousarray(audio))
        if in_sr != model_sr:
            wave_t = AF.resample(wave_t, in_sr, model_sr)
        enhanced = enhance(model, df_state, wave_t)
        if in_sr != model_sr:
            enhanced = AF.resample(enhanced, model_sr, in_sr)

        out = enhanced.cpu().numpy()
        out = np.clip(out, -1.0, 1.0)
        out_i16 = (out * 32767.0).astype(np.int16).T  # (samples, channels)
        if out_i16.shape[1] == 1:
            out_i16 = out_i16[:, 0]

        base, ext = os.path.splitext(wavfile)
        out_path = f"{base}_denoised{ext}"
        wav.write(out_path, in_sr, out_i16)
        print(f"Denoised {wavfile} -> {out_path} ({in_sr} Hz)")


def cmd_plot(args):
    paths = expand_paths(args.wavfiles)
    if not paths:
        raise SystemExit("No WAV files to plot.")

    for wavfile in paths:
        plot_wav_file(wavfile)
        if args.output:
            # One image per input: insert the stem before the extension.
            if len(paths) == 1:
                out = args.output
            else:
                base, ext = os.path.splitext(args.output)
                stem = os.path.splitext(os.path.basename(wavfile))[0]
                out = f"{base}_{stem}{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {out}")

    if not args.output:
        plt.show()


def cmd_plot_same(args):
    paths = expand_paths(args.wavfiles)
    if not paths:
        raise SystemExit("No WAV files to plot.")

    plot_wav_files_overlaid(paths)
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


def cmd_spectrogram(args):
    paths = expand_paths(args.wavfiles)
    if not paths:
        raise SystemExit("No WAV files to plot.")

    for wavfile in paths:
        spectrogram_wav_file(wavfile)
        if args.output:
            # One image per input: insert the stem before the extension.
            if len(paths) == 1:
                out = args.output
            else:
                base, ext = os.path.splitext(args.output)
                stem = os.path.splitext(os.path.basename(wavfile))[0]
                out = f"{base}_{stem}{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved spectrogram to {out}")

    if not args.output:
        plt.show()


# --------------------------------------------------------------------------- recording

def slugify(name):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "mic"


def list_audio_devices(audio):
    print("Available audio input devices:")
    print("-" * 50)
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"Device {i}: {info['name']}")
            print(f"  - Max input channels: {info['maxInputChannels']}")
            print(f"  - Default sample rate: {info['defaultSampleRate']}")
            print()


def list_output_devices(audio):
    print("Available audio output devices:")
    print("-" * 50)
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"Device {i}: {info['name']}")
            print(f"  - Max output channels: {info['maxOutputChannels']}")
            print(f"  - Default sample rate: {info['defaultSampleRate']}")
            print()


def resolve_mic_names(audio, device_indices, override):
    """Map each device index to a unique, filesystem-safe name."""
    if override:
        names = [n.strip() for n in override.split(",")]
        if len(names) != len(device_indices):
            raise SystemExit(
                f"--names has {len(names)} entries but {len(device_indices)} devices given")
    else:
        names = [slugify(audio.get_device_info_by_index(i)["name"]) for i in device_indices]

    seen = {}
    unique = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def resolve_mic_channels(audio, device_indices, spec):
    """Per-device channel count: --channels '1,2,1'. Default mono each."""
    if not spec:
        return [1] * len(device_indices)
    try:
        counts = [int(c) for c in spec.split(",")]
    except ValueError:
        raise SystemExit("--channels must be comma-separated integers, e.g. 1,2,1")
    if len(counts) != len(device_indices):
        raise SystemExit(f"--channels has {len(counts)} entries but {len(device_indices)} devices given")
    for dev, nch in zip(device_indices, counts):
        available = audio.get_device_info_by_index(dev)["maxInputChannels"]
        if nch < 1 or nch > available:
            raise SystemExit(f"device {dev} supports 1..{available} input channels, got {nch}")
    return counts


class MicRecorder(threading.Thread):
    def __init__(self, audio, sample_format, device_index, name, rate, nchannels):
        super().__init__(daemon=True)
        self.audio = audio
        self.sample_format = sample_format
        self.device_index = device_index
        self.name = name
        self.rate = rate
        self.nchannels = nchannels
        self.frames = []
        self._stop_event = threading.Event()
        self._stream = None
        self.error = None

    def open(self):
        self._stream = self.audio.open(
            format=self.sample_format,
            channels=self.nchannels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=RECORD_CHUNK_SIZE,
        )

    def run(self):
        try:
            while not self._stop_event.is_set():
                data = self._stream.read(RECORD_CHUNK_SIZE, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as exc:  # noqa: BLE001 - surface to main thread
            self.error = exc
        finally:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()

    def stop(self):
        self._stop_event.set()

    def save_channels(self, out_dir, prefix):
        """De-interleave to one mono WAV per channel. Returns list of filenames."""
        samples = np.frombuffer(b"".join(self.frames), dtype=np.int16)
        if self.nchannels > 1:
            samples = samples.reshape(-1, self.nchannels)
        files = []
        for c in range(self.nchannels):
            label = self.name if self.nchannels == 1 else f"{self.name}_ch{c}"
            filename = f"{prefix}_{label}.wav"
            channel = samples if self.nchannels == 1 else samples[:, c]
            with wave.open(os.path.join(out_dir, filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(RECORD_SAMPLE_WIDTH)
                wf.setframerate(self.rate)
                wf.writeframes(channel.tobytes())
            files.append(filename)
        return files


def cmd_record(args):
    try:
        import pyaudio
    except ImportError:
        raise SystemExit("The 'record' command requires pyaudio (pip install pyaudio).")

    audio = pyaudio.PyAudio()
    try:
        if args.list:
            list_audio_devices(audio)
            return
        if not args.devices:
            raise SystemExit("provide at least one device index, or use --list")

        mic_names = resolve_mic_names(audio, args.devices, args.names)
        mic_channels = resolve_mic_channels(audio, args.devices, args.channels)

        os.makedirs(args.out, exist_ok=True)
        recorders = [
            MicRecorder(audio, pyaudio.paInt16, dev, name, args.rate, nch)
            for dev, name, nch in zip(args.devices, mic_names, mic_channels)
        ]
        for rec in recorders:
            rec.open()

        print(f"Output:  {args.out}")
        print(f"Prefix:  {args.prefix}")
        print("Mics:    " + ", ".join(
            f"{idx}->{name}({nch}ch)"
            for idx, name, nch in zip(args.devices, mic_names, mic_channels)))
        print(f"Rate:    {args.rate} Hz")
        print("-" * 50)

        for rec in recorders:
            rec.start()
        print("Recording... press Enter to stop.")
        try:
            input()
        except KeyboardInterrupt:
            print()
        for rec in recorders:
            rec.stop()
        for rec in recorders:
            rec.join()

        for name, err in [(r.name, r.error) for r in recorders if r.error]:
            print(f"  ! {name}: {err}")

        saved = []
        for rec in recorders:
            saved.extend(rec.save_channels(args.out, args.prefix))
        print(f"Saved {len(saved)} track(s) to {args.out}:")
        for filename in saved:
            print(f"  {filename}")
    finally:
        audio.terminate()


# --------------------------------------------------------------------------- loopback

LOOPBACK_CHUNK_SIZE = 256  # small buffer keeps the round-trip latency low


def prompt_device(audio, kind):
    """Ask the user to pick a device index of the given kind ('input'/'output')."""
    key = "maxInputChannels" if kind == "input" else "maxOutputChannels"
    valid = [i for i in range(audio.get_device_count())
             if audio.get_device_info_by_index(i)[key] > 0]
    if not valid:
        raise SystemExit(f"No {kind} devices found.")

    if kind == "input":
        list_audio_devices(audio)
    else:
        list_output_devices(audio)

    default = (audio.get_default_input_device_info() if kind == "input"
               else audio.get_default_output_device_info())["index"]
    while True:
        try:
            raw = input(f"Select {kind} device index [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nAborted.")
        if not raw:
            return default
        try:
            idx = int(raw)
        except ValueError:
            print("  Enter a device index (an integer).")
            continue
        if idx not in valid:
            print(f"  {idx} is not an available {kind} device.")
            continue
        return idx


def remap_channels(samples, in_channels, out_channels):
    """Reshape interleaved int16 frames from in_channels to out_channels.

    Mono in / multi out fans the signal out to every speaker; multi in / mono out
    averages. Otherwise channels are copied positionally, padding with silence or
    dropping the extras.
    """
    if in_channels == out_channels:
        return samples
    frames = samples.reshape(-1, in_channels)
    if in_channels == 1:
        out = np.repeat(frames, out_channels, axis=1)
    elif out_channels == 1:
        out = frames.mean(axis=1, keepdims=True).astype(np.int16)
    else:
        out = np.zeros((frames.shape[0], out_channels), dtype=np.int16)
        n = min(in_channels, out_channels)
        out[:, :n] = frames[:, :n]
    return out.reshape(-1)


def cmd_loopback(args):
    try:
        import pyaudio
    except ImportError:
        raise SystemExit("The 'loopback' command requires pyaudio (pip install pyaudio).")

    audio = pyaudio.PyAudio()
    in_stream = out_stream = None
    try:
        in_dev = args.input if args.input is not None else prompt_device(audio, "input")
        out_dev = args.output if args.output is not None else prompt_device(audio, "output")

        in_info = audio.get_device_info_by_index(in_dev)
        out_info = audio.get_device_info_by_index(out_dev)
        if in_info["maxInputChannels"] < 1:
            raise SystemExit(f"Device {in_dev} has no input channels.")
        if out_info["maxOutputChannels"] < 1:
            raise SystemExit(f"Device {out_dev} has no output channels.")

        rate = args.rate or int(in_info["defaultSampleRate"])
        in_ch = args.in_channels or 1
        if in_ch > in_info["maxInputChannels"]:
            raise SystemExit(
                f"input device {in_dev} supports 1..{int(in_info['maxInputChannels'])} channels")
        out_ch = args.out_channels or min(max(in_ch, 2), int(out_info["maxOutputChannels"]))
        if out_ch > out_info["maxOutputChannels"]:
            raise SystemExit(
                f"output device {out_dev} supports 1..{int(out_info['maxOutputChannels'])} channels")

        print("-" * 50)
        print(f"In:   {in_dev} {in_info['name']} ({in_ch}ch)")
        print(f"Out:  {out_dev} {out_info['name']} ({out_ch}ch)")
        print(f"Rate: {rate} Hz, chunk {args.chunk} frames "
              f"({1000.0 * args.chunk / rate:.1f} ms)")
        print("-" * 50)

        try:
            in_stream = audio.open(
                format=pyaudio.paInt16, channels=in_ch, rate=rate, input=True,
                input_device_index=in_dev, frames_per_buffer=args.chunk)
            out_stream = audio.open(
                format=pyaudio.paInt16, channels=out_ch, rate=rate, output=True,
                output_device_index=out_dev, frames_per_buffer=args.chunk)
        except Exception as exc:  # noqa: BLE001 - device/format mismatch is common here
            raise SystemExit(
                f"Could not open the streams at {rate} Hz: {exc}\n"
                "Try --rate with a rate both devices support (e.g. 44100 or 48000).")

        gain = 10.0 ** (args.gain / 20.0) if args.gain else None
        print("Looping back... press Ctrl-C to stop.")
        try:
            while True:
                data = in_stream.read(args.chunk, exception_on_overflow=False)
                if gain is None and in_ch == out_ch:
                    out_stream.write(data, exception_on_underflow=False)
                    continue
                samples = np.frombuffer(data, dtype=np.int16)
                if gain is not None:
                    samples = np.clip(samples.astype(np.float32) * gain,
                                      -32768, 32767).astype(np.int16)
                samples = remap_channels(samples, in_ch, out_ch)
                out_stream.write(samples.tobytes(), exception_on_underflow=False)
        except KeyboardInterrupt:
            print("\nStopped.")
    finally:
        for stream in (in_stream, out_stream):
            if stream is not None:
                stream.stop_stream()
                stream.close()
        audio.terminate()


def main():
    parser = argparse.ArgumentParser(description="A toolbox for all things audio.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plot = sub.add_parser("plot", help="Plot the waveform of one or more WAV files.")
    p_plot.add_argument("wavfiles", nargs="+", help="Path(s) or glob pattern(s) to input WAV file(s)")
    p_plot.add_argument("-o", "--output", help="Save plot(s) to image file instead of showing them "
                                              "(stem appended per file when multiple match)")
    p_plot.set_defaults(func=cmd_plot)

    p_plot_same = sub.add_parser("plot-same", help="Plot all waveforms overlaid on a single subplot.")
    p_plot_same.add_argument("wavfiles", nargs="+", help="Path(s) or glob pattern(s) to input WAV file(s)")
    p_plot_same.add_argument("-o", "--output", help="Save plot to image file instead of showing it")
    p_plot_same.set_defaults(func=cmd_plot_same)

    p_spec = sub.add_parser("spectrogram", help="Plot the spectrogram of one or more WAV files.")
    p_spec.add_argument("wavfiles", nargs="+", help="Path(s) or glob pattern(s) to input WAV file(s)")
    p_spec.add_argument("-o", "--output", help="Save spectrogram(s) to image file instead of showing them "
                                               "(stem appended per file when multiple match)")
    p_spec.set_defaults(func=cmd_spectrogram)

    p_gain = sub.add_parser("gain", help="Apply a digital gain (dB) to WAV file(s).")
    p_gain.add_argument("wavfiles", nargs="+", help="Path(s) or glob pattern(s) to input WAV file(s)")
    p_gain.add_argument("-d", "--db", type=float, required=True,
                        help="Gain in decibels (e.g. 6 ~= 2x, -6 ~= 0.5x)")
    p_gain.set_defaults(func=cmd_gain)

    p_eq = sub.add_parser("gain-equalize",
                          help="Match each file's level to a reference (percentile-based).")
    p_eq.add_argument("reference", help="reference WAV file whose level others are matched to")
    p_eq.add_argument("wavfiles", nargs="+", help="Path(s) or glob pattern(s) to equalize")
    p_eq.add_argument("--percentile", type=float, default=90.0,
                      help="percentile of |amplitude| to match (default: 90)")
    p_eq.set_defaults(func=cmd_gain_equalize)

    p_dn = sub.add_parser("denoise",
                          help="Suppress environmental noise with DeepFilterNet3.")
    p_dn.add_argument("wavfiles", nargs="+", help="Path(s) or glob pattern(s) to denoise")
    p_dn.add_argument("-g", "--gain", type=float, default=0.0,
                      help="dB gain applied before the denoiser (in-memory only; not saved). "
                           "Useful to lift very quiet recordings above the noise floor.")
    p_dn.set_defaults(func=cmd_denoise)

    p_rec = sub.add_parser("record", help="Record from one or more microphones to an output dir.")
    p_rec.add_argument("devices", nargs="*", type=int,
                       help="input device indices to record from (see --list)")
    p_rec.add_argument("--list", action="store_true", help="list input devices and exit")
    p_rec.add_argument("--names", help="comma-separated mic names (one per device)")
    p_rec.add_argument("--channels",
                       help="comma-separated channel count per device, e.g. 1,2,1 "
                            "(default: mono each). Each channel is saved separately.")
    p_rec.add_argument("--rate", type=int, default=DEFAULT_RECORD_RATE, help="sample rate (Hz)")
    p_rec.add_argument("-o", "--out", default=".", help="output directory (created if missing)")
    p_rec.add_argument("-p", "--prefix", default="recording",
                       help="filename prefix for saved tracks (<prefix>_<mic>.wav)")
    p_rec.set_defaults(func=cmd_record)

    p_lb = sub.add_parser("loopback",
                          help="Route a live input device straight to an output device.")
    p_lb.add_argument("-i", "--input", type=int,
                      help="input device index (prompts interactively if omitted)")
    p_lb.add_argument("-o", "--output", type=int,
                      help="output device index (prompts interactively if omitted)")
    p_lb.add_argument("--rate", type=int,
                      help="sample rate (Hz); default: the input device's default rate")
    p_lb.add_argument("--in-channels", type=int, help="input channel count (default: 1)")
    p_lb.add_argument("--out-channels", type=int,
                      help="output channel count (default: stereo if supported)")
    p_lb.add_argument("--chunk", type=int, default=LOOPBACK_CHUNK_SIZE,
                      help=f"frames per buffer; lower is lower latency "
                           f"(default: {LOOPBACK_CHUNK_SIZE})")
    p_lb.add_argument("-g", "--gain", type=float, default=0.0,
                      help="dB gain applied to the routed signal")
    p_lb.set_defaults(func=cmd_loopback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
