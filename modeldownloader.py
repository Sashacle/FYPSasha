import os
import subprocess
import sys
import requests
import importlib.metadata as metadata
from pathlib import Path
from tqdm import tqdm
from packaging import version

def create_directory_if_not_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True)

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

def upgrade_tts_package():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "tts"])
    except Exception as e:
        print(f"An error occurred while upgrading TTS: {e}")
        print("Try installing the new version manually")
        print("pip install --upgrade tts")

def upgrade_stream2sentence_package():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "stream2sentence"])
    except Exception as e:
        print(f"An error occurred while upgrading Stream2sentence: {e}")
        print("Stream2sentence installing the new version manually")
        print("pip install --upgrade stream2sentence")

def check_tts_version():
    try:
        tts_version = metadata.version("tts")
        if version.parse(tts_version) < version.parse("0.21.1"):
            upgrade_tts_package()
    except metadata.PackageNotFoundError:
        print("TTS is not installed.")

def check_stream2sentence_version():
    try:
        tts_version = metadata.version("stream2sentence")
        if version.parse(tts_version) < version.parse("0.2.0"):
            upgrade_stream2sentence_package()
    except metadata.PackageNotFoundError:
        print("stream2sentence is not installed.")

def download_model(this_dir, model_version):
    base_path = this_dir / 'models'
    model_path = base_path / f'v{model_version}'
    files_to_download = {
         "config.json": f"https://huggingface.co/coqui/XTTS-v2/raw/v{model_version}/config.json",
         "model.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/v{model_version}/model.pth?download=true",
         "vocab.json": f"https://huggingface.co/coqui/XTTS-v2/raw/v{model_version}/vocab.json"
    }
    create_directory_if_not_exists(base_path)
    create_directory_if_not_exists(model_path)
    for filename, url in files_to_download.items():
         destination = model_path / filename
         if not destination.exists():
             print(f"[XTTS] Downloading {filename}...")
             download_file(url, destination)
