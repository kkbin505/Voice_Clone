import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# === 1. 读取音频 ===
wav_path = "output_audio/tts_survery.wav"     # 修改成你的文件名


# ======= 输入文件路径 =======
wav_original = "output_audio/original_audio.wav"   # 原始音频
wav_generated = "output_audio/tts_audio_1.wav" # 合成音频

# ======= 加载音频 =======
y1, sr1 = librosa.load(wav_original, sr=None)
y2, sr2 = librosa.load(wav_generated, sr=None)

# 保证采样率一致（否则 Mel 参数不一致）
assert sr1 == sr2, "两个音频采样率不一致，请先重采样！"
sr = sr1

# ======= 生成 Mel-spectrogram =======
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

# ======= 画 2×2 对比图 =======
plt.figure(figsize=(14, 8))

# --- 左上：原始波形 ---
plt.subplot(2, 2, 1)
librosa.display.waveshow(y1, sr=sr)
plt.title("Original - Waveform")

# --- 右上：合成波形 ---
plt.subplot(2, 2, 2)
librosa.display.waveshow(y2, sr=sr)
plt.title("Generated - Waveform")

# --- 左下：原始 Mel-spectrogram ---
plt.subplot(2, 2, 3)
librosa.display.specshow(mel1, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Original - Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

# --- 右下：合成 Mel-spectrogram ---
plt.subplot(2, 2, 4)
librosa.display.specshow(mel2, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Generated - Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.show()
