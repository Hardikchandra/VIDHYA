import speech_recognition as sr

def list_available_microphones():
    """Lists all available microphones and prints the device names."""
    print("Listing available microphones...")
    mic_names = sr.Microphone.list_microphone_names()
    if not mic_names:
        print("No microphones found.")
    for i, name in enumerate(mic_names):
        print(f"Device {i}: {name}")
    return mic_names

def recognize_speech_from_mic():
    """Recognizes speech from the microphone and handles different errors."""
    try:
        # Initialize recognizer
        r = sr.Recognizer()

        # Listing available microphones
        mic_names = list_available_microphones()

        # If no microphones are found, print an error and return
        if not mic_names:
            print("No microphones available. Exiting...")
            return

        # Use the correct device index based on the list above
        mic_index = 1  # Set to the correct index (Device 1: USB Audio Device)

        with sr.Microphone(device_index=mic_index) as source:
            if source is None:
                print("Error: Microphone source is not initialized correctly.")
                return

            print(f"Using microphone: {mic_names[mic_index]}")

            # Adjust for ambient noise (this step listens for silence/noise in the environment)
            print("Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=3)  # Increase duration if needed

            print("Listening for speech... Please speak something.")
            
            # Listen for speech (timeout and phrase_time_limit to prevent indefinite blocking)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)  # Adjust timeout and phrase_time_limit
            except sr.WaitTimeoutError:
                print("Timeout occurred while waiting for speech.")
                return
            except sr.RequestError as e:
                print(f"Request error: Could not request results from Google Speech Recognition service; {e}")
                return
            except KeyboardInterrupt:
                print("\nProgram interrupted. Exiting...")
                return
            except Exception as e:
                print(f"An error occurred during listening: {e}")
                return

            if audio is None:
                print("No audio recorded. It may have timed out waiting for speech.")
                return

            print("Listening done.")

            # Recognize speech using Google Web Speech API
            print("Recognizing speech...")
            try:
                result = r.recognize_google(audio)
                print(f"Recognized Speech: {result}")
                return result
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                print(f"An error occurred during recognition: {e}")

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to call the recognize speech function
if __name__ == "__main__":
    print("Starting speech recognition...")
    recognize_speech_from_mic()
