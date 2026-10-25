"""Generate the synthetic sEMG dataset used by the demo.

No participant data are used or redistributed. Every sample is produced by a
simple parametric model,

    s(t) = envelope(t) * sum_k a_k * sin(2 * pi * f_k * t + phi_k) + noise(t),

in which the frequencies ``f_k`` are drawn from a band that depends on the
synthetic diagnosis level. The resulting mean/median frequency therefore
decreases from level 0 to level 2, which gives the feature extraction,
clustering and PCA stages something to separate. This is a demonstration model
only: the numbers obtained from this data carry no clinical meaning.

Output layout, matching ``feature_analysis.data_loader``::

    demo/data/<action_type>/<subject_id>/<channel_id>.csv   # column EMG_Clean
    demo/metadata_clinical_diagnosis.csv                    # subject_id, diagnosis_level

Usage::

    python demo/make_demo_data.py            # write demo/data and the metadata
    python demo/make_demo_data.py --check    # verify the generated files
"""

import argparse
import os

import numpy as np
import pandas as pd

FS = 2000                                  # sampling rate, Hz
DURATION = 8.5                             # duration of one recording, s
BURSTS = [(0.80, 1.90), (5.20, 1.90)]      # (onset, duration) of muscle activity, s
N_COMPONENTS = 120                         # sinusoids per recording
NOISE_STD = 0.006                          # baseline noise, also present at rest

SUBJECTS = {"S01": 0, "S02": 1, "S03": 2}  # subject_id -> synthetic level
ACTIONS = ["bending", "sit-stand"]
CHANNELS = ["ch1", "ch2"]

# Level-dependent spectral content and amplitude.
CLASS_MODEL = {
    0: {"band": (70.0, 330.0), "amp": 0.85},   # normal
    1: {"band": (58.0, 250.0), "amp": 0.70},   # osteopenia
    2: {"band": (45.0, 175.0), "amp": 0.55},   # osteoporosis
}

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_NAME = "metadata_clinical_diagnosis.csv"


def _envelope(n_samples, fs=FS, ramp=0.12):
    """Return a burst envelope on a zero background.

    Each burst is gated with raised-cosine ramps of length ``ramp`` to avoid
    discontinuities at the edges.
    """
    env = np.zeros(n_samples)
    t = np.arange(n_samples) / fs

    for onset, duration in BURSTS:
        env[(t >= onset) & (t < onset + duration)] = 1.0
        for edge in (onset, onset + duration):
            edge_window = np.abs(t - edge) < ramp
            phase = (t[edge_window] - (edge - ramp)) / (2 * ramp)
            if edge == onset:
                env[edge_window] = 0.5 - 0.5 * np.cos(np.pi * phase)
            else:
                env[edge_window] = 0.5 + 0.5 * np.cos(np.pi * phase)

    return env


def synth_signal(level, seed, amp_scale=1.0, f_shift=0.0):
    """Return one synthetic recording for the given diagnosis level."""
    rng = np.random.default_rng(seed)
    n_samples = int(DURATION * FS)
    t = np.arange(n_samples) / FS

    f_low, f_high = CLASS_MODEL[level]["band"]
    f_low = max(f_low + f_shift, 5.0)
    f_high = max(f_high + f_shift, f_low + 10.0)

    freqs = rng.uniform(f_low, f_high, N_COMPONENTS)
    phases = rng.uniform(0, 2 * np.pi, N_COMPONENTS)
    weights = rng.uniform(0.4, 1.0, N_COMPONENTS)
    weights /= np.sqrt(np.sum(weights ** 2))          # approximately unit RMS

    carrier = np.zeros(n_samples)
    for freq, phase, weight in zip(freqs, phases, weights):
        carrier += weight * np.sin(2 * np.pi * freq * t + phase)

    amplitude = CLASS_MODEL[level]["amp"] * amp_scale
    signal = amplitude * _envelope(n_samples) * carrier
    signal += rng.normal(0.0, NOISE_STD, n_samples)
    return signal


def build(demo_dir=DEMO_DIR):
    """Write the csv files and the metadata table."""
    data_dir = os.path.join(demo_dir, "data")

    for action in ACTIONS:
        for subject_id, level in SUBJECTS.items():
            out_dir = os.path.join(data_dir, action, subject_id)
            os.makedirs(out_dir, exist_ok=True)

            for channel_index, channel in enumerate(CHANNELS):
                # Deterministic seed; hash() would be salted per process.
                seed = (ord(subject_id[1]) * 7919
                        + ACTIONS.index(action) * 104729
                        + channel_index * 1299709
                        + level * 15485863) % (2 ** 31 - 1)
                signal = synth_signal(
                    level,
                    seed=seed,
                    amp_scale=1.0 if channel_index == 0 else 0.9,
                    f_shift=0.0 if channel_index == 0 else -8.0,
                )
                pd.DataFrame({"EMG_Clean": signal}).to_csv(
                    os.path.join(out_dir, f"{channel}.csv"), index=False, float_format="%.4f"
                )

    metadata = pd.DataFrame({
        "subject_id": list(SUBJECTS),
        "diagnosis_level": list(SUBJECTS.values()),
        "sex": ["F", "F", "M"],
        "age": [58, 63, 71],
    })
    metadata.to_csv(os.path.join(demo_dir, METADATA_NAME), index=False)

    n_files = len(SUBJECTS) * len(ACTIONS) * len(CHANNELS)
    print(f"wrote {n_files} csv files to {data_dir}")
    print(f"wrote {os.path.join(demo_dir, METADATA_NAME)}")


def check(demo_dir=DEMO_DIR):
    """Verify the layout, column names and recording lengths."""
    data_dir = os.path.join(demo_dir, "data")
    n_files = 0

    for action in ACTIONS:
        for subject_id in SUBJECTS:
            for channel in CHANNELS:
                path = os.path.join(data_dir, action, subject_id, f"{channel}.csv")
                if not os.path.exists(path):
                    raise SystemExit(f"missing: {path}")
                recording = pd.read_csv(path)
                if "EMG_Clean" not in recording.columns:
                    raise SystemExit(f"{path}: no EMG_Clean column")
                if len(recording) != int(DURATION * FS):
                    raise SystemExit(f"{path}: unexpected length {len(recording)}")
                n_files += 1

    metadata = pd.read_csv(os.path.join(demo_dir, METADATA_NAME))
    levels = sorted(int(level) for level in metadata["diagnosis_level"].unique())
    print(f"ok: {n_files} files, {int(DURATION * FS)} samples each, "
          f"{len(metadata)} labelled subjects, levels={levels}")


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic demo data.")
    parser.add_argument("--demo-dir", default=DEMO_DIR)
    parser.add_argument("--check", action="store_true",
                        help="verify the generated files instead of rewriting them")
    args = parser.parse_args()

    if args.check:
        check(args.demo_dir)
    else:
        build(args.demo_dir)
        check(args.demo_dir)


if __name__ == "__main__":
    main()
