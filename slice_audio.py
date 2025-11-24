import os
import librosa
import soundfile as sf
import numpy as np

# ================= 配置区域 =================
# 输入音频文件夹
INPUT_FOLDER = "raw_audio"
# 输出切片文件夹
OUTPUT_FOLDER = "dataset_sliced"

# 【关键】针对 6GB 显存的优化参数
MIN_DURATION = 2.0   # 最短 2 秒 (太短容易是杂音)
MAX_DURATION = 10.0  # 最长 10 秒 (超过 12秒 6GB显存容易爆)
SILENCE_DB = 40      # 静音阈值 (分贝)，越小越灵敏
# ===========================================

def slice_audio():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    if not files:
        print(f"❌ 错误: 在 {INPUT_FOLDER} 文件夹里没找到音频文件！")
        return

    print(f"🔪 开始处理，检测到 {len(files)} 个文件...")
    
    total_saved = 0

    for file in files:
        file_path = os.path.join(INPUT_FOLDER, file)
        try:
            # 加载音频 (sr=44100 保证高音质)
            y, sr = librosa.load(file_path, sr=44100)
            
            # 去除静音片段，获取非静音区间
            # top_db=SILENCE_DB: 低于此分贝视为静音
            intervals = librosa.effects.split(y, top_db=SILENCE_DB)

            for i, (start, end) in enumerate(intervals):
                chunk = y[start:end]
                duration = len(chunk) / sr

                # 【核心逻辑】筛选符合长度的片段
                if MIN_DURATION <= duration <= MAX_DURATION:
                    # 保存文件
                    filename = f"{os.path.splitext(file)[0]}_{i:03d}.wav"
                    save_path = os.path.join(OUTPUT_FOLDER, filename)
                    sf.write(save_path, chunk, sr)
                    print(f"  ✅ 保存切片: {filename} ({duration:.2f}s)")
                    total_saved += 1
                else:
                    # 过长或过短的丢弃（为了显存安全）
                    pass
                    
        except Exception as e:
            print(f"  ❌ 处理文件 {file} 失败: {e}")

    print("-" * 30)
    print(f"🎉 处理完成！共生成 {total_saved} 个切片。")
    print(f"📂 请检查文件夹: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    slice_audio()