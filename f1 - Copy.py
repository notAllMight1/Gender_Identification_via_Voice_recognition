import speech_recognition as sr
import pyaudio
import numpy as np
import wave

# Constants
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
CHUNK_SIZE = 7505
MALE_THRESHOLD = 120  # Adjusted threshold
FEMALE_THRESHOLD = 220  # Adjusted threshold
NEUTRAL_ZONE_LOWER = 150
NEUTRAL_ZONE_UPPER = 200

def record_audio():
    audio = pyaudio.PyAudio()
    
    try:
        # Open an audio stream to capture audio from the microphone
        stream = audio.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        print("Recording...")
        audio_data = np.array([], dtype=np.int16)

        for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE)):
            audio_chunk = stream.read(CHUNK_SIZE)
            audio_data = np.append(audio_data, np.frombuffer(audio_chunk, dtype=np.int16))

        print("Recording finished.")

        # Save the audio data as a WAV file
        save_audio(audio_data, "audio.wav")

        return audio_data

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

def classify_gender(audio_data):
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    pitch = SAMPLE_RATE / np.argmax(autocorr)
    
    if pitch < MALE_THRESHOLD:
        print("Male voice detected")
    elif pitch > FEMALE_THRESHOLD:
        print("Female voice detected")
    else:
        if NEUTRAL_ZONE_LOWER <= pitch <= NEUTRAL_ZONE_UPPER:
            print("Voice gender undetermined (neutral)")
        else:
            print("Voice gender unclear")

def main():
    try:
        audio_data = record_audio()
        recognize_speech("audio.wav")
        classify_gender(audio_data)
    except Exception as e:
        print("An error occurred: {0}".format(e))

if __name__ == "__main__":
    main()
