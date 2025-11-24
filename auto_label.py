import os
import whisper
import logging

# ================= CONFIGURATION =================
# Path to the folder containing sliced audio files
AUDIO_FOLDER = "dataset_sliced"

# Path to the output text file (training list)
OUTPUT_FILE = "filelist.txt"

# Name of the speaker (customizable)
SPEAKER_NAME = "Jack"

# Language of the audio ('ZH' for Chinese, 'EN' for English, 'JA' for Japanese)
# Note: GPT-SoVITS uses 'ZH', 'EN', 'JA' as language tags.
LANGUAGE = "EN" 

# Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'
# 'small' is recommended for 6GB VRAM. It balances speed and accuracy.
MODEL_SIZE = "small"
# =================================================

def generate_labels():
    # Check if the audio folder exists
    if not os.path.exists(AUDIO_FOLDER):
        print(f"Error: Audio folder '{AUDIO_FOLDER}' not found.")
        return

    print(f"--- Loading Whisper model: {MODEL_SIZE} ---")
    try:
        # Load the model onto the GPU
        model = whisper.load_model(MODEL_SIZE, device="cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: Make sure your GPU is available (torch.cuda.is_available()).")
        return

    print("--- Model loaded. Starting transcription... ---")

    # Get list of wav files
    files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")]
    files.sort() # Sort files to keep order

    results = []
    
    # Open the output file to write results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for idx, filename in enumerate(files):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            
            try:
                # Transcribe audio
                # beam_size=5 improves accuracy slightly
                result = model.transcribe(file_path, beam_size=5)
                text = result["text"].strip()

                # Format for GPT-SoVITS: path|speaker|language|text
                # We use the absolute path to avoid path issues during training
                abs_path = os.path.abspath(file_path)
                line = f"{abs_path}|{SPEAKER_NAME}|{LANGUAGE}|{text}"
                
                # Write to file and memory
                results.append(line)
                f_out.write(line + "\n")
                f_out.flush() # Ensure data is written to disk immediately

                print(f"[{idx+1}/{len(files)}] {filename} -> {text}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print("-" * 30)
    print(f"Done! Processed {len(results)} files.")
    print(f"Labels saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    # Filter warnings to keep the output clean
    logging.getLogger("whisper").setLevel(logging.WARNING)
    generate_labels()