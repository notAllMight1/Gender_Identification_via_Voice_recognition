import speech_recognition as sr
import pyaudio
import numpy as np
import wave
import threading
from genderize import Genderize

# Constants
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
MALE_THRESHOLD = 135  # Adjusted threshold
FEMALE_THRESHOLD = 250  # Adjusted threshold
NEUTRAL_ZONE_LOWER = 135
NEUTRAL_ZONE_UPPER = 185

# Global variable to store audio data
audio_data = None

# Global variable to signal when to stop recording
stop_recording = False

def record_audio():
    global audio_data
    audio = pyaudio.PyAudio()
    
    try:
        # Open an audio stream to capture audio from the microphone
        stream = audio.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        print("Recording... (Press 'q' and Enter to stop)")
        audio_data = np.array([], dtype=np.int16)

        while not stop_recording:
            audio_chunk = stream.read(CHUNK_SIZE)
            audio_data = np.append(audio_data, np.frombuffer(audio_chunk, dtype=np.int16))

        print("Recording finished.")

    except Exception as e:
        print("An error occurred during audio recording: {0}".format(e))
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def save_audio(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio format
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

def recognize_speech(filename):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(filename) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))

def predict_speaker_name():
    speaker_name = input("Enter the speaker's name: ")
    return speaker_name

def classify_gender(audio_data):
    # Calculate the mean pitch of the recorded audio
    mean_pitch = np.mean(audio_data)
    print(f"Mean Pitch: {mean_pitch} Hz")

    # Use the calculated pitch to classify the gender
    if mean_pitch >= MALE_THRESHOLD:
        gender = "Male"
    elif mean_pitch <= FEMALE_THRESHOLD:
        gender = "Female"
    else:
        gender = "Neutral"

    return gender

def main():
    global stop_recording
    try:
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()

        input("Press Enter to stop recording...")

        stop_recording = True
        recording_thread.join()

        if audio_data is not None:
            save_audio(audio_data, 'D:/misc/Programming/python projects/speech/audio.wav')
            recognize_speech("audio.wav")

            # Predict the speaker's name
            speaker_name = predict_speaker_name()
            print(f"Speaker's name: {speaker_name}")

            # Call the classify_gender function with the recorded audio data
            predicted_gender = classify_gender(audio_data)
            print(f"Predicted gender based on voice: {predicted_gender}")
        else:
            print("No audio data available.")

    except Exception as e:
        print("An error occurred: {0}".format(e))

if __name__ == "__main__":
    main()

