import os
import subprocess
import platform
from gtts import gTTS
import re

def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Cleans input text of markdown formatting and asterisks,
    prints it, then uses gTTS to convert to speech.
    """
    # Clean markdown-style asterisks (*, **) used for bullets and bold
    clean_text = re.sub(r'\*+', '', input_text)

    # Print for debugging
    print("\n🧼 Cleaned text sent to TTS:\n")
    print(clean_text)
    print("\n" + "-" * 60 + "\n")

    # Generate TTS
    tts = gTTS(text=clean_text.strip(), lang='en')
    tts.save(output_filepath)

    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":
            subprocess.run(['mpg123', output_filepath])
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")
