# Vlyra

> An AI-powered music assistant that listens to performances, understands musical input, and provides feedback to musicians.

Vlyra is an open-source artificial intelligence project focused on helping musicians improve by analysing audio performances. It uses machine learning and audio processing techniques to recognise musical information and provide useful feedback.

## Features

* Audio analysis and feature extraction
* Note recognition from instrument recordings
* Machine learning models for music understanding
* Performance evaluation and feedback generation
* Research-friendly training and evaluation pipeline

## How It Works

Vlyra processes audio through several stages:

```
Audio Input
     |
     v
Audio Preprocessing
     |
     v
Feature Extraction
     |
     v
AI Model
     |
     v
Musical Analysis & Feedback
```

The system converts raw audio into meaningful representations, analyses the musical content using trained models, and produces insights about the performance.

## Project Goals

The long-term goal of Vlyra is to create an intelligent music companion that can help musicians practise more effectively.

Possible future features:

* Chord recognition
* Rhythm and timing analysis
* Pitch accuracy feedback
* Expression and dynamics analysis
* Support for multiple instruments
* Real-time performance feedback

## Technologies

* Python
* PyTorch / machine learning frameworks
* Librosa for audio processing
* NumPy
* Scientific computing tools

## Project Structure

```
vlyra/
├── src/              # Main application code
└── README.md
└── requirements.txt
```

## Models
Click [here](https://simarpreetsingh.org/downloads/vlyra/latest/vlyra-latest.pt) to download the latest version. Please note that vlyra is in extremely early development and does not yet have much functionality.

## Dataset

Vlyra uses publicly available music datasets for training and evaluation.

Datasets are not included in this repository due to size and licensing restrictions. Instructions for obtaining and preparing datasets can be found in the documentation.

## Running Vlyra

Clone the repository:

```bash
git clone https://github.com/simarpreet/vlyra.git
cd vlyra
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference:

```bash
python predict.py example.wav
```

## Development

Contributions are welcome. If you would like to improve Vlyra, you can:

* Add new audio analysis methods
* Improve model architectures
* Add support for new instruments
* Improve documentation
* Create new evaluation methods

## Roadmap

* [x] Initial audio processing pipeline
* [x] Basic note recognition experiments
* [ ] Improved model accuracy
* [ ] Multi-instrument support
* [ ] Real-time analysis
* [ ] Music feedback system

## License

Vlyra is licensed under the Apache License 2.0.

See `LICENSE` for more information.

## Author

Created by Simarpreet Singh.
