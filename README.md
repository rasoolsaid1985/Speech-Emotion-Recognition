
# Speech Recognition using LSTM

This repository contains the code for building a speech recognition model using Long Short-Term Memory (LSTM) networks. The model is trained using MFCC features extracted from audio files and is designed to classify speech emotions.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Introduction
Speech recognition has a wide range of applications, from voice assistants to dictation software. This project demonstrates how to use LSTM networks for recognizing different emotions in speech using Python's Keras and Librosa libraries.

## Installation
To get started, clone the repository and install the necessary dependencies.

### Requirements

Make sure you have Python 3 installed, then install the following packages:

```bash
pip install numpy pandas seaborn matplotlib librosa tensorflow
```

If you're using a Windows environment, you can install Librosa using:

```bash
pip install librosa
```

### Additional Requirements
Librosa may require additional packages like `ffmpeg` or `soundfile`. You can install them with:

```bash
pip install soundfile
```

## Dataset
The dataset used for this project contains audio files labeled according to different emotions (e.g., neutral, happy, sad). You can replace it with any other audio dataset, provided that the files are organized in a similar structure.

## Usage

1. Clone this repository.
2. Place your dataset in the appropriate folder or modify the code to point to your dataset.
3. Run the notebook or script to extract features, train the model, and visualize the results.

### Running the Model

You can run the model training by executing the notebook or Python script provided in the repository.

```bash
python train_model.py
```

You can also explore the notebook `speech_recognition_lstm.ipynb` for step-by-step instructions and code explanations.

## Model Architecture

The model consists of an LSTM layer followed by fully connected dense layers for classification. The audio files are pre-processed using MFCC (Mel Frequency Cepstral Coefficients) to extract relevant features from the speech.

```python
model = Sequential([
    LSTM(123, return_sequences=False, input_shape=(40,1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
```

## Results
After training for 100 epochs, the model achieves satisfactory results in recognizing different emotions in speech. You can visualize the training and validation loss by using the provided plotting function.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
