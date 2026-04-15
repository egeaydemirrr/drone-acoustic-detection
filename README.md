# Drone Acoustic Detection System

An audio-based drone detection and classification system that identifies drones by their acoustic signature using mel-spectrograms and a CNN. Built as a defense/security ML project — the idea is that drones produce distinct harmonic patterns from their propellers, and a model can learn to spot these even in noisy environments.

## What it does

- Detects whether a drone is present in a 1-second audio clip
- Classifies between two drone types (Bebop, Mambo) and non-drone sounds
- Tested under different noise levels (SNR 0-20dB) to simulate real-world conditions

## Results

| Condition | Accuracy | Drone Recall |
|-----------|----------|--------------|
| Clean     | 97.1%    | 97.0%        |
| 20dB SNR  | 93.4%    | 95.5%        |
| 10dB SNR  | 88.9%    | 91.5%        |
| 5dB SNR   | 84.9%    | 87.0%        |
| 0dB SNR   | 79.1%    | 81.5%        |

## How to run

1. Open `Drone_Detection.ipynb` in Google Colab
2. Select Runtime → Change runtime type → T4 GPU
3. Run all cells — dataset downloads automatically (~200 MB)

## Tech stack

- Python, PyTorch, librosa
- Mel-spectrogram feature extraction
- CNN with 3 conv layers, early stopping, class weights
- SNR-based robustness evaluation

## Dataset

[DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset) by Sara Al-Emadi (IWCMC 2019) — 11,704 audio samples across 3 classes (Bebop drone, Mambo drone, non-drone environmental sounds).

## Future work

- Scale to DADS dataset (180k samples, 15+ drone models)
- Test with real-world noise (wind, traffic) instead of white noise
- Try Audio Spectrogram Transformer architecture
