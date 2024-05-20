# PolyGlot Speak
"Polyglot Speak" is an innovative translation system designed to bridge language barriers in education and intercultural communication by translating videos from Russian to English while preserving the original speaker's unique voice and facial expressions. This approach ensures that the speaker's emotions and intentions are maintained, overcoming common limitations of traditional translation methods. Primarily aimed at supporting remote communities in Kyrgyzstan, "Polyglot Speak" enhances access to education and fosters a richer intercultural exchange through its reliable translation model and user-friendly interface.
# Demo
[Google Drive Video](https://drive.google.com/file/d/1IjpMTz0mpn2GwcQhtL0wzikH7LbW5rlF/view?usp=drive_link)
# Install 
this project uses CUDA and runs on a graphic card
To install project use these steps:
- ### git clone https://github.com/Sashacle/FYPSasha
- ### cd PolyGlotSpeaks
- ### python -m venv venv
- ### venv/scripts/activate
- ### pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu 118 --index-url https://download.pytorch.org/whl/cu118
- ### pip install -r requerments.txt
# How to use
After installing the required dependencies and models (Whisper and XTTS), follow these steps:
1. Run the `gui.py` file.
2. Once the GUI application is launched, input the following parameters:
   - For the "Enter mode", input `1`.
   - For the "target language", input `"ru"`.
   - For the "selected language", input `"en"`.
3. Select the input folder and output folder.
4. Execute the code.
