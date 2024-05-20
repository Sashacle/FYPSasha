import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, PhotoImage
import torch
from moviepy.editor import VideoFileClip, AudioFileClip

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")

def overlay_audio_on_video(video_path, audio_path, output_video_path):
    clear_cuda_cache()
    print("Cleared CUDA cache")
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', bitrate='8000k')

def run_processing():
    clear_cuda_cache()

    input_path = filedialog.askopenfilename(title="Select Video or Audio File",
                                            filetypes=[("Media Files", "*.mp4 *.mkv *.avi *.wav *.mp3"),
                                                       ("All Files", "*.*")])
    if not input_path:
        messagebox.showerror("Error", "A media file must be selected!")
        return

    mode = mode_entry.get()
    target_lang = target_lang_entry.get()
    speaker_lang = speaker_lang_entry.get()
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    if not all([mode, target_lang, speaker_lang, output_folder]):
        messagebox.showerror("Error", "All fields must be filled!")
        return

    command = [
        "python", "main.py",
        "--input", input_path,
        "--mode", mode,
        "--target_lang", target_lang,
        "--speaker_lang", speaker_lang,
        "--output_folder", output_folder
    ]

    try:
        subprocess.run(command, check=True)
        audio_path = os.path.join(output_folder, "result.mp3")
        video_output_path = os.path.join(output_folder, "result_video.mp4")
        overlay_audio_on_video(input_path, audio_path, video_output_path)
        messagebox.showinfo("Success", "Processing completed successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Polyglot Speak - Media Processing")
root.geometry("700x600")
image_path = r"D:\Final_Project\PolyglotSpeaks\.venv\logo.png"
image = PhotoImage(file=image_path)
image = image.subsample(3, 3)
image_label = tk.Label(root, image=image)
image_label.pack(pady=10)
mode_label = tk.Label(root, text="Enter mode (1, 2, 3):")
mode_label.pack()
mode_entry = tk.Entry(root)
mode_entry.pack()
target_lang_label = tk.Label(root, text="Enter target language code (e.g., 'ru'):")
target_lang_label.pack()
target_lang_entry = tk.Entry(root)
target_lang_entry.pack()
speaker_lang_label = tk.Label(root, text="Enter speaker language code (e.g., 'en'):")
speaker_lang_label.pack()
speaker_lang_entry = tk.Entry(root)
speaker_lang_entry.pack()
input_button = tk.Button(root, text="Select Video/Audio File", command=run_processing)
input_button.pack(pady=10)
output_button = tk.Button(root, text="Select Output Folder", command=lambda: filedialog.askdirectory(title="Select Output Folder"))
output_button.pack(pady=10)
process_button = tk.Button(root, text="Run Processing", command=run_processing)
process_button.pack(pady=10)
root.mainloop()
