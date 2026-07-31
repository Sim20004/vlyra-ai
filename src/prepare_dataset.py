import json
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


# Smaller than 256 to avoid unnecessary memory usage
SPECTROGRAM_SIZE = (192, 192)

DATASET_ROOT = Path(
    "/home/simarpreetsingh/music_ai/dataset"
)

OUTPUT_ROOT = Path(
    "processed"
)

OUTPUT_ROOT.mkdir(
    exist_ok=True
)


def create_spectrogram(wav_path):

    audio, sr = librosa.load(
        wav_path,
        sr=16000
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=SPECTROGRAM_SIZE[0],
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalise 0-1
    mel_db -= mel_db.min()

    maximum = mel_db.max()

    if maximum > 0:
        mel_db /= maximum

    # Ensure fixed width
    mel_db = librosa.util.fix_length(
        mel_db,
        size=SPECTROGRAM_SIZE[1],
        axis=1
    )

    # Crop if needed
    mel_db = mel_db[
        :SPECTROGRAM_SIZE[0],
        :SPECTROGRAM_SIZE[1]
    ]

    return mel_db.astype(
        np.float16
    )


def process_split(split_name):

    split_dir = (
        DATASET_ROOT /
        split_name
    )

    audio_dir = (
        split_dir /
        "audio"
    )

    print(
        f"\nProcessing {split_name}"
    )

    with open(
        split_dir /
        "examples.json"
    ) as f:
        metadata = json.load(f)


    keyboard_notes = [
        (name, info)
        for name, info in metadata.items()
        if info["instrument_family_str"]
        == "keyboard"
    ]


    print(
        f"Found {len(keyboard_notes)} keyboard samples"
    )


    # Allocate memory once
    X = np.zeros(
        (
            len(keyboard_notes),
            SPECTROGRAM_SIZE[0],
            SPECTROGRAM_SIZE[1]
        ),
        dtype=np.float16
    )

    y = np.zeros(
        len(keyboard_notes),
        dtype=np.int64
    )


    count = 0


    for note_name, info in tqdm(
        keyboard_notes,
        desc=split_name
    ):

        wav_path = (
            audio_dir /
            f"{note_name}.wav"
        )


        if not wav_path.exists():
            continue


        try:

            X[count] = create_spectrogram(
                wav_path
            )

            y[count] = info["pitch"]

            count += 1


        except Exception as e:

            print(
                f"\nFailed {note_name}: {e}"
            )


    # Remove unused space
    X = X[:count]
    y = y[:count]


    print(
        f"{split_name}: {count} samples"
    )

    print(
        "Shape:",
        X.shape
    )


    np.save(
        OUTPUT_ROOT /
        f"X_{split_name}.npy",
        X
    )

    np.save(
        OUTPUT_ROOT /
        f"y_{split_name}.npy",
        y
    )


    print(
        f"Saved X_{split_name}.npy"
    )

    print(
        f"Saved y_{split_name}.npy"
    )


def main():

    process_split(
        "train"
    )

    process_split(
        "valid"
    )

    process_split(
        "test"
    )

    print(
        "\nFinished"
    )


if __name__ == "__main__":
    main()