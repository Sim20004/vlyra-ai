import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ======================
# Configuration
# ======================

BATCH_SIZE = 16

DATASET_ROOT = Path(
    "../dataset"
)

TEST_ROOT = (
    DATASET_ROOT /
    "test"
)

MODEL_PATH = Path(
    "best_model.pt"
)

PITCH_MAP_PATH = Path(
    "pitch_to_class.json"
)

SPECTROGRAM_SIZE = (
    192,
    192
)


# ======================
# Spectrogram creation
# ======================

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


    mel_db -= mel_db.min()

    maximum = mel_db.max()

    if maximum > 0:

        mel_db /= maximum


    mel_db = librosa.util.fix_length(
        mel_db,
        size=SPECTROGRAM_SIZE[1],
        axis=1
    )


    mel_db = mel_db[
        :SPECTROGRAM_SIZE[0],
        :SPECTROGRAM_SIZE[1]
    ]


    return mel_db.astype(
        np.float32
    )



# ======================
# Dataset
# ======================

class NSynthTestDataset(Dataset):

    def __init__(
        self,
        root,
        pitch_to_class
    ):

        self.samples = []


        with open(
            root /
            "examples.json"
        ) as f:

            metadata = json.load(f)



        skipped = 0


        for name, info in metadata.items():


            if (
                info["instrument_family_str"]
                != "keyboard"
            ):
                continue



            pitch = info["pitch"]



            if pitch not in pitch_to_class:

                skipped += 1

                continue



            self.samples.append(

                (

                    root /
                    "audio" /
                    f"{name}.wav",

                    pitch_to_class[pitch]

                )

            )



        print(
            f"Loaded {len(self.samples)} test samples"
        )


        if skipped:

            print(
                f"Skipped {skipped} samples with unknown pitches"
            )



    def __len__(self):

        return len(
            self.samples
        )



    def __getitem__(
        self,
        index
    ):

        wav_path, label = (
            self.samples[index]
        )


        spec = create_spectrogram(
            wav_path
        )


        spec = torch.tensor(
            spec,
            dtype=torch.float32
        )


        spec = spec.unsqueeze(
            0
        )


        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return spec, label



# ======================
# Model
# ======================

class NoteRecognizer(nn.Module):

    def __init__(
        self,
        classes
    ):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )



        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),


            nn.Flatten(),


            nn.Linear(
                128,
                256
            ),


            nn.ReLU(),


            nn.Dropout(
                0.3
            ),


            nn.Linear(
                256,
                classes
            )

        )



    def forward(
        self,
        x
    ):

        x = self.features(x)

        return self.classifier(x)



# ======================
# Testing
# ======================

def test():

    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        "Device:",
        device
    )



    with open(
        PITCH_MAP_PATH
    ) as f:

        pitch_to_class = json.load(f)



    # JSON keys are strings
    pitch_to_class = {

        int(k): v

        for k, v in pitch_to_class.items()

    }



    classes = len(
        pitch_to_class
    )


    print(
        "Classes:",
        classes
    )



    dataset = NSynthTestDataset(

        TEST_ROOT,

        pitch_to_class

    )



    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )



    model = NoteRecognizer(
        classes
    )


    print(
        "Loading model..."
    )


    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=device

        )

    )


    model.to(device)

    model.eval()



    correct = 0

    total = 0



    print(
        "Testing..."
    )



    with torch.no_grad():

        for X, y in tqdm(loader):

            X = X.to(device)

            y = y.to(device)



            output = model(X)


            predictions = output.argmax(
                dim=1
            )


            correct += (

                predictions == y

            ).sum().item()



            total += len(y)



    accuracy = (

        correct /

        total *

        100

    )



    print()

    print(
        "========== RESULTS =========="
    )

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )



if __name__ == "__main__":

    test()