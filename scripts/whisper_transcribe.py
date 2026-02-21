"""Batch Whisper transcription for MFA alignment."""
import os
import sys
import glob
from tqdm import tqdm

def main():
    wav_dir = "/data1/yangzhuowei/MambaTalk/data/wave16k"
    out_dir = "/data1/yangzhuowei/MambaTalk/data/mfa_input"
    
    os.makedirs(out_dir, exist_ok=True)
    
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    print(f"Found {len(wav_files)} wav files")
    
    # Check how many already have txt
    existing = set()
    for f in glob.glob(os.path.join(out_dir, "*.txt")):
        existing.add(os.path.splitext(os.path.basename(f))[0])
    
    todo = [f for f in wav_files if os.path.splitext(os.path.basename(f))[0] not in existing]
    print(f"Already done: {len(existing)}, remaining: {len(todo)}")
    
    if len(todo) == 0:
        print("All files already transcribed!")
        return
    
    import whisper
    model = whisper.load_model("base")
    
    for wav_path in tqdm(todo, desc="Transcribing"):
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            result = model.transcribe(wav_path, language="en")
            text = result["text"].strip()
            if not text:
                text = "."
        except Exception as e:
            print(f"Error on {basename}: {e}")
            text = "."
        
        # Write txt to mfa_input dir
        with open(os.path.join(out_dir, f"{basename}.txt"), "w") as f:
            f.write(text)
        
        # Also symlink the wav to mfa_input dir
        wav_link = os.path.join(out_dir, f"{basename}.wav")
        if not os.path.exists(wav_link):
            os.symlink(wav_path, wav_link)
    
    print("Done!")

if __name__ == "__main__":
    main()
