import os
from gtts import gTTS
import tempfile
import pygame
from langdetect import DetectorFactory

# Ensure consistent language detection results
DetectorFactory.seed = 0

pygame.mixer.init()

def convert_numbers_and_terms_to_hindi(text):
    number_map = {
        '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
        '5': 'पाँच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ'
    }
    
    term_map = {
        'x': 'गुणा',  # multiplication
        '*': 'गुणा',  # multiplication (if used)
        '+': 'जोड़',   # addition
        '-': 'घटाना',  # subtraction
        '/': 'भाग',    # division
        '=': 'बराबर'   # equal
    }
    
    # Replace numbers
    for eng_num, hindi_num in number_map.items():
        text = text.replace(eng_num, hindi_num)
    
    # Replace mathematical terms
    for eng_term, hindi_term in term_map.items():
        text = text.replace(eng_term, hindi_term)
    
    return text

def is_num(text):
    num_characters = set("123456789")
    return any(char in num_characters for char in text)
    
def speak_text(text):
    temp_audio_file_name = None

    try:
        if is_num(text)== False:
            print("Recognized English text, no processing needed.")
            lang = 'en'
            tts = gTTS(text=text, lang=lang, slow=False)
        else:
            print("Assuming text is in Hindi.")
            text = convert_numbers_and_terms_to_hindi(text)
            lang = 'hi'
            tts = gTTS(text=text, lang=lang, slow=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file_name = temp_audio_file.name
            tts.save(temp_audio_file_name)

        pygame.mixer.music.load(temp_audio_file_name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"An error occurred during text-to-speech processing: {e}")

    finally:
        if temp_audio_file_name and os.path.exists(temp_audio_file_name):
            os.remove(temp_audio_file_name)

def stop_music():
    pygame.mixer.music.stop()
    print("Music stopped.")

if __name__ == "__main__":
    print("Enter the text to speak (e.g., '1 + 2 = 3'). Type 'exit' to quit.")
    while True:
        text_to_speak = input("Input: ")
        if text_to_speak.lower() == 'exit':
            print("Exiting the program.")
            break
        
        speak_text(text_to_speak)
