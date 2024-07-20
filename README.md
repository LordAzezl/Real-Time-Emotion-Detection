# Real-Time Emotion Detection 

## Overview

This project aims to detect human emotions in real-time using a webcam feed and a pre-trained deep learning model. The detected emotion is then used to suggest songs based on the emotion detected.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Model Training](#model-training)
6. [Running the Emotion Detection](#running-the-emotion-detection)
7. [Spotify Integration](#spotify-integration)
8. [Contributions](#contributions)
9. [License](#license)

## Prerequisites

- Python 3.x
- TensorFlow and Keras
- OpenCV
- NumPy
- Spotipy
- Tkinter (for GUI)
- Haar Cascade file for face detection
- Pre-trained emotion detection model (`model_optimal.h5`)

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/LordAzezl/Real-Time-Emotion-Detection.git
cd Real-Time-Emotion-Detection
```

2. **Install required Python packages**:

```bash
pip install numpy tensorflow opencv-python spotipy
```

3. **Download necessary files**:

- Download the Haar Cascade file for face detection and place it in the project directory.
- Ensure you have the pre-trained model file (`model_optimal.h5`) in the project directory.

## Usage

### Running the Emotion Detection

To run the real-time emotion detection, use the following command:

```bash
python emotion_detection.py
```

### Spotify Integration

To use the Spotify integration, you need to have a Spotify developer account. Create a new application on the Spotify Developer Dashboard and obtain the `clientID`, `clientSecret`, and `redirect_uri`. These credentials should be placed in the appropriate section of the script.

## Project Structure

```
Real-Time-Emotion-Detection/
│
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── model_optimal.h5  # Pre-trained model for emotion detection
├── SongsList/  # Directory containing CSV files with song lists
│   ├── EngHappy.csv
│   ├── EngSad.csv
│   ├── HindiHappy.csv
│   ├── HindiSad.csv
│   └── Allsongs.csv
├── emotion_detection.py  # Main script for emotion detection and Spotify integration
├── model_training.py  # Script for training the emotion detection model
└── README.md  # This README file
```

## Model Training

### Data Preparation

- The training and validation datasets should be placed in appropriate directories (`train` and `test` respectively).
- The images should be grayscale and of size 48x48 pixels.

### Training the Model

The `model_training.py` script contains the code for training the emotion detection model. To train the model, execute the script:

```bash
python model_training.py
```

This script uses convolutional neural networks (CNNs) to train the model on the provided dataset and saves the trained model as `model_optimal.h5`.

### Model Evaluation

The script evaluates the model on the training and validation datasets and prints the final training and validation accuracy.

## Running the Emotion Detection

The `emotion_detection.py` script captures video from the webcam, detects faces, and predicts the emotion of each detected face in real-time. Based on the detected emotion, it selects a random song from the appropriate CSV file and plays it using the Spotify API.

```bash
python emotion_detection.py
```

## Spotify Integration

The script uses the `spotipy` library to interact with the Spotify API. It authenticates the user and retrieves their profile information. When an emotion is detected, it searches for a song from the corresponding CSV file and opens it in the web browser using Spotify.

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License.

---

This README provides an overview of the Real-Time Emotion Detection project, including installation instructions, usage examples, and a description of the code structure.
