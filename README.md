# Chord Recognition with Deep Learning

## Project Overview

This project implements a deep learning model to recognize and classify musical chords from audio features. Chord recognition is a fundamental task in music information retrieval (MIR) with applications in automatic transcription, music analysis, and intelligent music production systems. The ability to automatically identify chords from audio enables musicians, producers, and music analysts to quickly understand harmonic structures without manual annotation.

In this project, we:

1.  Process and analyze the McGill Billboard chord dataset
2.  Develop and optimize a hybrid CNN-LSTM neural network architecture
3.  Evaluate different model configurations through extensive hyperparameter tuning
4.  Create a robust chord recognition system that can identify chords from chroma features

## Why Chord Recognition Matters

Chord recognition is vital for:

* **Music Education**: Helping students learn music theory and chord progressions
* **Music Production**: Enabling automatic chord detection for digital audio workstations
* **Music Analysis**: Supporting musicological research and computational music theory
* **Music Recommendation**: Improving music similarity measures based on harmonic content

## Dataset Description

### The McGill Billboard Dataset

The McGill Billboard Dataset is a comprehensive collection of chord annotations for popular music. It consists of annotations for songs drawn from the Billboard "Hot 100" charts between 1958 and 1991, providing a diverse representation of popular music across different eras.

**Citation**: John Ashley Burgoyne, Jonathan Wild, and Ichiro Fujinaga, ‘An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis’, in Proceedings of the 12th International Society for Music Information Retrieval Conference, ed. Anssi Klapuri and Colby Leider (Miami, FL, 2011), pp. 633–38

[https://ddmal.music.mcgill.ca/research/The\_McGill\_Billboard\_Project\_(Chord\_Analysis\_Dataset)/](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/)

**Dataset Characteristics**:

* Annotations for 890 song excerpts
* Time-aligned chord labels with detailed harmonic information
* Corresponding audio features (chroma vectors) for each song

The dataset provides both chord annotations (in the form of lab files) and chroma features, which are 12-dimensional vectors representing the intensity of each pitch class (C, C#, D, etc.) in short time windows.

## Data Loading and Initial Exploration

The project begins with loading the dataset index and performing initial data cleaning. We then explore the dataset's temporal characteristics, such as the distribution of songs across decades. We verify the availability of chroma features and chord labels, ensuring data integrity.

## Analyzing Chord Distribution Across the Dataset

Understanding the distribution of chord labels is crucial. We analyze chord frequencies to identify class imbalances and gain insights into the harmonic content of the dataset. This analysis informs decisions about handling rare chords and feature engineering.

## Feature Engineering: Chord Simplification

The dataset contains detailed chord annotations. For practical recognition purposes, we simplify the chord vocabulary while preserving essential harmonic information. This involves mapping complex chord notations to a reduced set of chord types (e.g., major, minor, dominant 7th). We analyze the impact of this simplification on the chord distribution and coverage.

## Data Processing Pipeline

We define a data processing pipeline to prepare the dataset for training:

1.  **Feature Extraction**: Extract 12-dimensional chroma vectors for each time frame.
2.  **Time Alignment**: Align chord labels with chroma frames.
3.  **Windowing**: Create overlapping windows of chroma frames to capture temporal context.
4.  **Chord Simplification**: Reduce the chord vocabulary to a manageable set of chord types.
5.  **Train-Validation-Test Split**: Perform a stratified split to ensure that all chord types are represented in each set.

## Model Architecture

We implement a hybrid CNN-LSTM neural network for chord recognition. This architecture leverages both the spatial pattern recognition capabilities of CNNs and the temporal modeling capabilities of LSTMs, which is ideal for music audio analysis.

The architecture includes:

1.  **Convolutional layers**: Extract features from the chroma input
2.  **Bottleneck layer**: Reduce dimensionality and focus on essential features
3.  **Residual connections**: Improve gradient flow and prevent vanishing gradients
4.  **Bidirectional LSTM layers**: Model temporal dependencies in both directions
5.  **Attention mechanism**: Focus on the most relevant parts of the input sequence
6.  **Dense layers**: Final classification

### Understanding the CNN-LSTM Architecture for Chord Recognition

The model architecture for chord recognition combines convolutional neural networks (CNNs) and long short-term memory (LSTM) networks in a hybrid design that leverages the strengths of both approaches.

**Input Layer**

The input to our model consists of windowed chroma features with shape `(window_size, 12)`, where:

* `window_size` is the number of time frames.
* `12` represents the 12 pitch classes.

**Convolutional Feature Extraction**

The first stage of our model applies convolutional layers to extract meaningful patterns from the chroma features. These layers perform 1D convolutions across the time dimension with multiple filter sizes, allowing the model to learn different temporal patterns. We use multiple filter scales, kernel sizes, batch normalization, and ReLU activation.

**Bottleneck Layer (Optional)**

A bottleneck layer can be included to reduce dimensionality and focus the model on the most essential features. This is implemented as a 1×1 convolution.

**Residual Connections (Optional)**

Residual connections help address the vanishing gradient problem and enable training of deeper networks. These connections allow gradients to flow directly from later layers to earlier ones.

**Bidirectional LSTM Layers**

After extracting features with convolutions, we use bidirectional LSTM layers to model temporal dependencies in both directions. LSTMs are particularly effective for this task because they can remember long-term dependencies across the time dimension.

**Attention Mechanism (Optional)**

An attention mechanism allows the model to focus on the most relevant parts of the input sequence. The attention mechanism computes an attention score for each time step using a dense layer, applies a softmax to get a probability distribution over time steps, weights the LSTM outputs according to these attention scores, and combines the weighted outputs into a context vector.

**Dense Classification Layers**

The final stage of the model consists of fully connected layers for classification. These layers include dropout and batch normalization for regularization and end with a softmax activation to produce a probability distribution over chord classes.

**Model Compilation**

The model is compiled with categorical cross-entropy loss and the Adam optimizer.

**Architecture Benefits**

This hybrid CNN-LSTM architecture offers several advantages for chord recognition:

1.  Hierarchical Feature Learning
2.  Bidirectional Context
3.  Attention Mechanism
4.  Regularization
5.  Residual Connections

## Hyperparameter Tuning

Through extensive experimentation, we identified optimal hyperparameters. This configuration balances model capacity with generalization capability, achieving the best validation accuracy in our hyperparameter search. We implemented a hyperparameter tuning function that explores different combinations of hyperparameters, trains a model with each combination, evaluates its performance, and identifies the best set of parameters.

## Model Evaluation

Our deep learning-based chord recognition system was trained on the McGill Billboard dataset. The model achieves 81.7% training accuracy and 63.4% validation accuracy, with a best validation loss of 1.625 at epoch 32.

### Accuracy and Loss Dynamics

The accuracy and loss curves demonstrate typical deep learning training behavior with some important characteristics. The increasing gap between training and validation metrics indicates that the model is memorizing training examples rather than learning generalizable patterns.

### Learning Rate Adaptation

The model employed adaptive learning rate reduction, which was triggered by plateaus in validation loss.

### Early Stopping Effectiveness

The early stopping mechanism correctly identified that validation performance peaked and stopped training, preventing unnecessary computation and potential further overfitting.

## Conclusions and Future Work

Our chord recognition model demonstrates the viability of deep learning for this challenging music information retrieval task.

1.  **Addressing Overfitting**: Future work should explore more advanced regularization techniques or increased dataset size.
2.  **Architectural Improvements**: Newer architectures like Transformers could potentially capture long-range dependencies in music more effectively.
3.  **Feature Engineering**: Incorporating additional musical features beyond chroma vectors could provide complementary information.
4.  **Data Augmentation**: Implementing music-specific augmentation strategies could improve generalization.
