import requests
import json
import time
import threading
import webrtcvad
import sounddevice as sd
import numpy as np
from audio_out import speak_text
from speech_rec import recognize_speech_from_mic
from recognizer import recognize_faces_in_video
import pyttsx3  # For TTS functionality
from playsound import playsound  # For playing sound on interruption
import csv  # To log to CSV file
import os
from dotenv import load_dotenv

# Configurations
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
INACTIVITY_TIMEOUT = 10
EXIT_KEYWORDS = ["goodbye", "गुडबाय", "अलविदा"]
RESTART_KEYWORDS = ["stop", "स्टॉप"]
ACTIVATION_KEYWORDS = ["hello", "हेलो"]

# Global flag for speech interruption
interrupt_speaking = False
vad = webrtcvad.Vad(0)  # Initial VAD sensitivity

# Initialize pyttsx3 engine for speech synthesis
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate as needed

# Set Hindi male voice if available
voices = engine.getProperty('voices')
male_voice_set = False
for voice in voices:
    if "Hindi" in voice.name or "hi" in voice.id:
        if "male" in voice.name.lower():  # Check for male voice
            engine.setProperty('voice', voice.id)
            male_voice_set = True
            break

if not male_voice_set:
    print("Hindi male voice not available. Using the default male voice.")

def greet_user(name):
    greeting = f"Hello {name}"
    print(greeting)
    speak_with_interrupt(greeting)

def get_speech_input():
    
    user_input = recognize_speech_from_mic()
    print(f"User Input: {user_input}")
    return user_input or ""

def call_api(input_text):
    #preprompt= "Please respond in Hindi"
    #final_prompt = preprompt +input_text
    if not API_KEY:
        print("Missing GOOGLE_API_KEY. Set it in your environment or .env file.")
        return None
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}'
    payload = {"contents": [{"parts": [{"text": input_text}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status code {response.status_code}: {response.text}")
        return None

def process_response(result):
    try:
        # Check if 'candidates' and the necessary keys exist in the result
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    content = parts[0]['text'].replace('**', '').strip()
                    return ' '.join(content.split()[:200])  # Ensure response length is controlled
                else:
                    print("No 'text' found in 'parts'.")
            else:
                print("No 'content' or 'parts' found in 'candidate'.")
        else:
            print("No valid candidates found in the response.")
    except KeyError as e:
        print(f"Error processing response: Missing key {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Return a default message or handle the error case
    return "Sorry, there was an issue processing the response."

def log_to_csv(user_input, response):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
    with open('speech_output.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write a new row with timestamp, user input, and the generated response
        writer.writerow([timestamp, user_input, response])
    print("Logged user input and response to 'speech_output.csv'.")

def check_keywords(user_input, keywords):
    return any(keyword in user_input.lower() for keyword in keywords)

def listen_for_activation():
    print("Listening for activation...")
    while True:
        user_input =recognize_speech_from_mic()
        if user_input and check_keywords(user_input, ACTIVATION_KEYWORDS):
            print("Activation keyword detected.")
            return True

def speak_with_interrupt(text):
    global interrupt_speaking
    interrupt_speaking = False

    # Start speaking the text
    speak_text(text)

    # Short delay to let `speak_text` begin before checking for interruptions
    time.sleep(0.5)

    # Start the interrupt thread to detect user input
    stop_thread = threading.Thread(target=check_for_interrupt)
    stop_thread.start()

    # Set interrupt flag once the thread has started
    interrupt_speaking = True

    # Wait for the interrupt thread to finish
    stop_thread.join()

def check_for_interrupt():
    global interrupt_speaking
    vad.set_mode(2)  # Lower sensitivity temporarily while speaking
    sustained_detections = 0  # Counter for sustained voice detections

    while not interrupt_speaking:
        if detect_user_voice():
            sustained_detections += 1
            # Require multiple detections for sustained interruption
            if sustained_detections >= 3:
                stop_speaking()  # Stop if sustained user voice detected
                interrupt_speaking = True
                print("Stopped speaking due to user input.")
                break
        else:
            sustained_detections = 0  # Reset counter if no voice is detected

        time.sleep(0.05)
    
    vad.set_mode(0)  # Reset VAD sensitivity after speaking ends

def get_audio_chunk(duration_ms=30, sample_rate=16000):
    try:
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        audio_data = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        audio_data_bytes = audio_data.tobytes()
        if len(audio_data_bytes) != num_samples * 2:
            print("Captured audio chunk is not the expected length.")
            return None
        return audio_data_bytes
    except Exception as e:
        print(f"Error capturing audio chunk: {e}")
        return None

def detect_user_voice():
    audio_data = get_audio_chunk()
    if audio_data:
        is_speech = vad.is_speech(audio_data, sample_rate=16000)
        print(f"Voice detected: {is_speech}")
        return is_speech
    return False

def stop_speaking():
    engine.stop()  # Stop ongoing speech
    sound_file_path = 'interrupt.wav'
    try:
        playsound(sound_file_path)  # Play the interruption sound
    except Exception as e:
        print(f"Error playing sound: {e}")
    print("Speech stopped.")

def main():
    print("3")
    while True:
        if listen_for_activation():
            recognized_name = recognize_faces_in_video()
            if recognized_name:
                greet_user(recognized_name)
                start_time = time.time()

                while True:
                    print("Taking user input")
                    user_input = get_speech_input()
                    print("Taken user input")
                    if user_input and check_keywords(user_input, EXIT_KEYWORDS):
                        print("Goodbye!")
                        return
                    if user_input and check_keywords(user_input, RESTART_KEYWORDS):
                        print("Restarting...")
                        break

                    start_time = time.time()

                    if time.time() - start_time > INACTIVITY_TIMEOUT:
                        print("No input detected for 10 seconds. Closing the program.")
                        return

                    # Call API with user input
                    print("Calling API")
                    result = call_api(user_input)
                    print("Getted data from API")
                    if result:
                        generated_response = process_response(result)
                        print("Generated response:", generated_response)
                        speak_with_interrupt(generated_response)
                        log_to_csv(user_input, generated_response)  # Log the user input and response
            else:
                print("No face recognized.")

if __name__ == "__main__":
    print("1")
    # Create a header in the CSV file if it doesn't exist
    with open('speech_output.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # Check if file is empty (for header)
            writer.writerow(['Timestamp', 'User Input', 'Response'])
    print("2")
    main()
