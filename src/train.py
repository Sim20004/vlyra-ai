import json
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


MAX_RAM_PERCENT = 90
BATCH_SIZE = 16
EPOCHS = 40000


class SpectrogramDataset(Dataset):

    def __init__(
        self,
        x_path,
        y_path,
        pitch_to_class
    ):

        self.X = np.load(
            x_path,
            mmap_mode="r"
        )

        pitches = np.load(
            y_path
        )

        self.y = np.array(
            [
                pitch_to_class[
                    int(pitch)
                ]
                for pitch in pitches
            ],
            dtype=np.int64
        )

        self.normalise = (
            self.X.dtype == np.uint8
        )

    def __len__(self):

        return len(
            self.y
        )

    def __getitem__(
        self,
        index
    ):

        x = self.X[index]

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        if self.normalise:
            x = x / 255.0

        x = x.unsqueeze(0)

        y = torch.tensor(
            self.y[index],
            dtype=torch.long
        )

        return x, y


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


def memory_check():

    ram = psutil.virtual_memory()

    if (
        ram.percent >=
        MAX_RAM_PERCENT
    ):

        print()
        print(
            f"RAM usage reached "
            f"{ram.percent}%"
        )

        print(
            "Stopping safely."
        )

        sys.exit(1)


def build_pitch_mapping():

    y = np.load(
        "processed/y_train.npy"
    )

    pitches = sorted(
        set(
            y.tolist()
        )
    )

    pitch_to_class = {

        pitch: index

        for index,
        pitch

        in enumerate(
            pitches
        )
    }

    class_to_pitch = {

        index: pitch

        for index,
        pitch

        in enumerate(
            pitches
        )
    }

    return (
        pitch_to_class,
        class_to_pitch
    )


def evaluate(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in loader:

            memory_check()

            X = X.to(device)
            y = y.to(device)

            output = model(X)

            predictions = (
                output.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == y
            ).sum().item()

            total += len(y)

    return (
        correct /
        total *
        100
    )


def main():

    processed = Path(
        "processed"
    )

    print(
        "Creating pitch mapping..."
    )

    (
        pitch_to_class,
        class_to_pitch
    ) = build_pitch_mapping()

    with open(
        "pitch_to_class.json",
        "w"
    ) as f:

        json.dump(
            pitch_to_class,
            f,
            indent=4
        )

    with open(
        "class_to_pitch.json",
        "w"
    ) as f:

        json.dump(
            class_to_pitch,
            f,
            indent=4
        )

    classes = len(
        pitch_to_class
    )

    print(
        "Classes:",
        classes
    )

    print(
        "Loading datasets..."
    )

    train_set = SpectrogramDataset(
        processed /
        "X_train.npy",

        processed /
        "y_train.npy",

        pitch_to_class
    )

    valid_set = SpectrogramDataset(
        processed /
        "X_valid.npy",

        processed /
        "y_valid.npy",

        pitch_to_class
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    model = NoteRecognizer(
        classes
    ).to(device)

    if Path(
        "best_model.pt"
    ).exists():

        print(
            "Loading existing "
            "best_model.pt..."
        )

        model.load_state_dict(
            torch.load(
                "best_model.pt",
                map_location=device
            )
        )

        print(
            "Loaded successfully"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = (
        nn.CrossEntropyLoss()
    )

    best = evaluate(
        model,
        valid_loader,
        device
    )

    print(
        f"Starting validation: "
        f"{best:.2f}%"
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0

        progress = tqdm(
            train_loader,
            desc=
            f"Epoch "
            f"{epoch+1}"
            f"/{EPOCHS}"
        )

        for X, y in progress:

            memory_check()

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(X)

            loss = loss_fn(
                output,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=
                f"{loss.item():.4f}"
            )

        average_loss = (
            total_loss /
            len(
                train_loader
            )
        )

        accuracy = evaluate(
            model,
            valid_loader,
            device
        )

        print()
        print(
            f"Loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Validation: "
            f"{accuracy:.2f}%"
        )

        torch.save(
            model.state_dict(),
            "checkpoint.pt"
        )

        if accuracy > best:

            best = accuracy

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

            print(
                "Saved best model"
            )

    print()
    print(
        f"Best validation: "
        f"{best:.2f}%"
    )

    print(
        "Finished"
    )


if __name__ == "__main__":
    main()
