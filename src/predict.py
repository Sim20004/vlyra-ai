import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class NoteRecognizer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def midi_to_note(midi):
    note = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return f"{note}{octave}"


def create_spectrogram(wav_file):
    audio, _ = librosa.load(wav_file, sr=None)

    spectrogram = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=1.0
    )

    spectrogram -= spectrogram.min()

    if spectrogram.max() > 0:
        spectrogram /= spectrogram.max()

    spectrogram *= 255

    image = Image.fromarray(
        spectrogram.astype(np.uint8)
    )

    image = image.resize(
        (128, 128),
        Image.Resampling.LANCZOS
    )

    return np.array(image, dtype=np.float32)


def main():
    print("Loading model...")

    with open("class_to_pitch.json") as f:
        class_to_pitch = json.load(f)

    num_classes = len(class_to_pitch)

    model = NoteRecognizer(num_classes)

    model.load_state_dict(
        torch.load(
            "note_model.pt",
            map_location="cpu"
        )
    )

    model.eval()

    print("Loading validation metadata...")

    with open("../dataset/valid/examples.json") as f:
        metadata = json.load(f)

    correct = 0
    total = 0

    errors = []

    worst = []

    confusions = Counter()

    for note_name, info in tqdm(metadata.items()):
        wav_file = (
            Path("../dataset/valid/audio")
            / f"{note_name}.wav"
        )

        spectrogram = create_spectrogram(wav_file)

        spectrogram /= 255.0

        tensor = torch.tensor(
            spectrogram,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)

            predicted_class = (
                torch.argmax(output, dim=1)
                .item()
            )

        predicted_pitch = int(
            class_to_pitch[
                str(predicted_class)
            ]
        )

        actual_pitch = info["pitch"]

        error = abs(
            predicted_pitch
            - actual_pitch
        )

        errors.append(error)

        if predicted_pitch == actual_pitch:
            correct += 1
        else:
            confusions[
                (
                    actual_pitch,
                    predicted_pitch
                )
            ] += 1

        worst.append(
            (
                error,
                actual_pitch,
                predicted_pitch,
                note_name
            )
        )

        total += 1

    errors = np.array(errors)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Accuracy: {100 * correct / total:.2f}%")
    print(f"Correct : {correct}")
    print(f"Wrong   : {total - correct}")
    print(f"Total   : {total}")

    print()

    print(
        "Average semitone error:",
        round(errors.mean(), 3)
    )

    print(
        "Median semitone error:",
        round(np.median(errors), 3)
    )

    print()

    print(
        "Within 1 semitone:",
        round(
            (errors <= 1).mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Within 2 semitones:",
        round(
            (errors <= 2).mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Within 12 semitones:",
        round(
            (errors <= 12).mean() * 100,
            2
        ),
        "%"
    )

    print()

    print("=" * 60)
    print("20 WORST ERRORS")
    print("=" * 60)

    worst.sort(reverse=True)

    for (
        error,
        actual,
        predicted,
        name
    ) in worst[:20]:

        print(
            f"{name}\n"
            f"  {actual:3d} {midi_to_note(actual)}"
            f" -> "
            f"{predicted:3d} {midi_to_note(predicted)}"
            f"  ({error} semitones)\n"
        )

    print("=" * 60)
    print("TOP CONFUSIONS")
    print("=" * 60)

    for (
        actual,
        predicted
    ), count in confusions.most_common(20):

        print(
            f"{midi_to_note(actual)}"
            f" -> "
            f"{midi_to_note(predicted)}"
            f" : {count}"
        )


if __name__ == "__main__":
    main()