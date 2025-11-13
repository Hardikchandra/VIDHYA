import requests
import json
import time
import threading
import webrtcvad
import sounddevice as sd
import numpy as np
from out_new import speak_text , stop_music
from speech_rec import recognize_speech_from_mic
from face2 import  recognize_face , capture_image
import pyttsx3
import csv
import tkinter as tk
from tkinter import Text, Label , Button
from PIL import Image, ImageTk, ImageSequence
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configurations
API_KEY = os.getenv("GOOGLE_API_KEY")
EXIT_KEYWORDS = ["goodbye", "good bye", "गुडबाय", "अलविदा"]
RESTART_KEYWORDS = ["restart", "clear all"]
ACTIVATION_KEYWORDS = ["hello", "hello vidhya","हेलो","hey"]

# Global flag for speech interruption
interrupt_speaking = False
vad = webrtcvad.Vad(0)  # Initial VAD sensitivity
response_cache = {}
log_queue = []
log_lock = threading.Lock()

# Initialize pyttsx3 engine for speech synthesis
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
male_voice_set = False
for voice in voices:
    if "Hindi" in voice.name or "hi" in voice.id:
        if "male" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            male_voice_set = True
            break

if not male_voice_set:
    print("Hindi male voice not available. Using the default male voice.")

def display_gif(label, path):
    """
    Displays a GIF on a tkinter label in a loop. The GIF restarts automatically 
    when it reaches the end.
    """
    img = Image.open(path)
    frames = ImageSequence.Iterator(img)

    # Terminate any previous animation
    if hasattr(label, "after_id") and label.after_id:
        label.after_cancel(label.after_id)
        label.after_id = None

    def animate():
        nonlocal frames
        try:
            frame = next(frames)
        except StopIteration:
            # Restart the frames iterator when the GIF ends
            frames = ImageSequence.Iterator(img)
            frame = next(frames)

        frame_image = ImageTk.PhotoImage(frame.convert("RGBA"))
        label.config(image=frame_image)
        label.image = frame_image  # Prevent garbage collection
        label.after_id = label.after(50, animate)  # Schedule next frame

    animate()

# Set up GUI window
window = tk.Tk()
window.title("V.I.D.H.Y.A")
window.geometry("2079x1134")
window.configure(bg="black")


# Load the background image
bg_image = Image.open("bg2.png")  # Replace with your image path
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label to hold the background image
bg_label = Label(window, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)


# Frame for dialogue box and buttons (horizontal alignment)
dialogue_frame = tk.Frame(window, bg="white")
dialogue_frame.pack(side="bottom", fill="x", padx=40, pady=10)

# Left Button
restart_button = tk.Button(
    dialogue_frame, 
    text="Re-Start", 
    command=lambda: restart_assistant(), 
    font=("Arial", 14, "bold"), 
    bg="green4", 
    fg="white", 
    activebackground="green3", 
    activeforeground="white", 
    padx=24, 
    pady=29, 
    relief="flat"
)
restart_button.pack(side="left", padx=10)

# Dialogue Box
dialogue_box = Text(
    dialogue_frame, 
    height=4, 
    wrap="word", 
    bg="white", 
    font=("Helvetica", 13, "bold"),
    bd=0, 
    relief="solid"
)
dialogue_box.pack(side="left", fill="x", expand=True)
# Center-align the text
dialogue_box.tag_configure("center", justify="center")

# Button
button = tk.Button(
    dialogue_frame, 
    text="Stop", 
    command=lambda: stop_assistant(), 
    font=("Arial", 14, "bold"), 
    bg="red", 
    fg="white", 
    activebackground="red2", 
    activeforeground="white", 
    padx=24, 
    pady=29, 
    relief="flat"
)
button.pack(side="right", padx=10)


# Avatar Display
avatar_label = Label(window, bg="white")
avatar_label.pack(side="bottom", pady=10, anchor="s")



# Auto-scroll Function
def insert_with_scroll(text):
    dialogue_box.insert(tk.END, text + "\n")
    dialogue_box.yview(tk.END)  
    window.update()  # Ensure immediate GUI refresh

# Core Assistant Functions


def stop_assistant():
    stop_music()
    print("Stopped Speaking")
    insert_with_scroll("Stopped Speaking")
    return True

def restart_assistant():
    insert_with_scroll("Assistant: Restarting...\n")
    # You can set a global flag or directly call the assistant_routine
    # For simplicity, we will just break the inner loop in the assistant_routine
    # This will be handled in the assistant_routine itself
    stop_music();
    global restart_flag
    restart_flag = True


def listen_for_activation():
    display_gif(avatar_label, "sleep.gif")
    dialogue_box.insert(tk.END, "Assistant: Listening for activation...\n")
    window.update()  
 
    while True:
        user_input = recognize_speech_from_mic(1)
        if user_input and check_keywords(user_input, ACTIVATION_KEYWORDS):
            dialogue_box.insert(tk.END, "Assistant: Activation keyword detected.\n")
            window.update()  # Update the GUI immediately
            return True

def greet_user(name):
    greeting = f"Hello {name} ! I am Vidhya How May i help you?"
    print(greeting)
    insert_with_scroll(f"Assistant: {greeting}\n")
    #display_gif(avatar_label, "home/vidhya/Desktop/final_proj/hello.gif")
    speak_with_interrupt(greeting)
    
    
    
def get_speech_input():
    display_gif(avatar_label, "hearing.gif")
    user_input = recognize_speech_from_mic(2)
    print(f"User  Input: {user_input}")
    return user_input or ""

def speak_with_interrupt(text):
    insert_with_scroll(f"Vidhya: {text}\n")
    greet_text="How May i help you?"
    if greet_text in text:
        display_gif(avatar_label, "hello.gif")
    else:
        display_gif(avatar_label, "speaking.gif")
        avatar_label.after(3000)
    speak_text(text)

  
    

# Additional Functions
ACTIVATION_KEYWORDS_SET = set(kw.lower() for kw in ACTIVATION_KEYWORDS)
EXIT_KEYWORDS_SET = set(kw.lower() for kw in EXIT_KEYWORDS)
RESTART_KEYWORDS_SET = set(kw.lower() for kw in RESTART_KEYWORDS)

def check_keywords(user_input, keywords_set):
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in keywords_set)

def call_api(input_text):
    if input_text in response_cache:
        return response_cache[input_text]
    if not API_KEY:
        print("Missing GOOGLE_API_KEY. Set it in your environment or .env file.")
        return None
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}'
    payload = {"contents": [{"parts": [{"text": input_text}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        response_cache[input_text] = result
        return result
    else:
        print(f"API request failed with status code {response.status_code}: {response.text}")
        return None

def process_response(result):
    try:
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    content = parts[0]['text'].replace('*', '').strip()
                    return ' '.join(content.split()[:60])  # Limit response length to 200 words
        print("No valid candidates found in the response.")
    except KeyError as e:
        print(f"Error processing response: Missing key {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return "Sorry, there was an issue processing the response."

def log_to_csv(user_input, response):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with log_lock:
        log_queue.append([timestamp, user_input, response])

def flush_logs_to_csv():
    while True:
        time.sleep(5)
        with log_lock:
            if log_queue:
                with open('speech_output.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(log_queue)
                    log_queue.clear()

log_thread = threading.Thread(target=flush_logs_to_csv, daemon=True)
log_thread.start()

restart_flag = False  # Global flag to indicate restart

def assistant_routine():
    global restart_flag  # Access the global flag
    while True:  # Outer loop for restarting the assistant
        restart_flag = False  # Reset the flag at the start of each routine
        if listen_for_activation():
            recognized_name = recognize_face(capture_image())
            insert_with_scroll(f"Recognized: {recognized_name}")
            if recognized_name:
                greet_user(recognized_name)

            while True:  # Inner loop for the main assistant routine
                insert_with_scroll("Listening.....")
                user_input = get_speech_input()
                
                # Check for restart immediately after getting user input
                if restart_flag:
                    insert_with_scroll("Assistant: Restarting...\n")
                    break  # Break the inner loop to restart

                if user_input:
                    print(f"User  said: {user_input}")
                    insert_with_scroll(f":User  {user_input}\n")

                    if check_keywords(user_input, EXIT_KEYWORDS_SET):
                        insert_with_scroll("Assistant: Goodbye!\n")
                        window.destroy()  # Close the window
                        print("Exiting")
                        return  # Exit the outer loop

                    elif check_keywords(user_input, RESTART_KEYWORDS_SET):
                        insert_with_scroll("Assistant: Restarting...\n")
                        restart_flag = True  # Set the flag to restart
                        break  # Break the inner loop to restart

                    result = call_api(user_input)
                    if result:
                        response = process_response(result)
                        print("Generated response:", response)
                        speak_with_interrupt(response)
                        log_to_csv(user_input, response)

                # Add a short delay before listening again
                time.sleep(1)  # Adjust the delay as needed
                
# Add the initialization block
if __name__ == "__main__":
    with open('speech_output.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(['Timestamp', 'User   Input', 'Response'])

    assistant_thread = threading.Thread(target=assistant_routine, daemon=True)
    assistant_thread.start()

    window.mainloop()
