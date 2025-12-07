import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# === 1. Read Audio ===
wav_path = "output_audio/tts_survery.wav"   


# ======= Input File =======
wav_original = "output_audio/original_audio.wav"   
wav_generated = "output_audio/tts_audio_1.wav" 

# ======= Load Audio =======
y1, sr1 = librosa.load(wav_original, sr=None)
y2, sr2 = librosa.load(wav_generated, sr=None)

# Make sure sample rate is the same
assert sr1 == sr2, "Please re sample to the same！"
sr = sr1

# ======= Mel-spectrogram =======
def get_mel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=0,
        fmax=sr/2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

mel1 = get_mel(y1, sr)
mel2 = get_mel(y2, sr)

# ======= Draw =======
plt.figure(figsize=(14, 8))

# --- Original Wave ---
plt.subplot(2, 2, 1)
librosa.display.waveshow(y1, sr=sr)
plt.title("Original - Waveform")

# --- Cloned Wave ---
plt.subplot(2, 2, 2)
librosa.display.waveshow(y2, sr=sr)
plt.title("Generated - Waveform")

plt.subplot(2, 2, 3)
librosa.display.specshow(mel1, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Original - Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.subplot(2, 2, 4)
librosa.display.specshow(mel2, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Generated - Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.show()
