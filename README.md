# Acoustic Analysis of African Tonal Languages for Speech Recognition

## Project carried out by: 
Jasmine GBOLOGANGBE
Joella CAÏTANO
Léonce LISBOA
Orphéric AMOUSSOU

## Project Overview

This project investigates the acoustic characteristics of African tonal languages and their impact on Automatic Speech Recognition (ASR) systems.

Many modern ASR models such as Whisper and wav2vec 2.0 were primarily trained on Indo-European languages. However, many African languages are tonal, meaning that pitch (F0) carries lexical or grammatical meaning. This raises an important question:

Do common acoustic representations (MFCC, Log-Mel spectrograms) capture tonal information sufficiently?

To explore this question, this project analyzes three African languages with different tonal characteristics:

- **Fongbé** – strongly tonal
- **Wolof** – weakly tonal
- **Swahili** – non-tonal

The project includes acoustic feature extraction, statistical analysis of tonal properties, and evaluation of ASR systems on these languages.

---

## Objectives

The project is structured around three main objectives:

1. **Acoustic Feature Analysis**
   - Waveform visualization
   - STFT spectrogram
   - Log-Mel spectrogram
   - MFCC extraction
   - F0 (pitch) contour analysis

2. **Quantitative Characterization of Tonal Features**
   - Statistical analysis of pitch (F0)
   - MFCC distribution analysis
   - PCA visualization
   - Language classification using ML models (SVM / KNN)

3. **Impact on Automatic Speech Recognition**
   - Transcription using Whisper models
   - Evaluation with WER and CER
   - Qualitative error analysis

---

## Dataset Sources

Audio samples are collected from:

- Mozilla Common Voice
- Google FLEURS Dataset
- Optional personal recordings for Fongbé

Each utterance is documented with metadata including:

- transcription
- translation
- speaker
- recording conditions

---

## Tech Stack

Python ecosystem:

- **librosa** – audio feature extraction
- **parselmouth** – pitch extraction
- **matplotlib** – visualization
- **numpy / scipy** – numerical analysis
- **scikit-learn** – ML classification
- **transformers** – Whisper ASR
- **datasets** – dataset loading
- **jiwer** – WER/CER evaluation

---

## Project Structure
