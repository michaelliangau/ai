# """
# Stream from a microphone into SeamlessM4T ASR model
# """
import pyaudio
import providers.seamlessm4t as seamlessm4t
import pyaudio
import wave
import time

# Define the stream parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5        # Duration to record
WAVE_OUTPUT_FILENAME = "tmp.wav"  # Temporary output WAV file

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a WAV file for writing
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)

# Define the callback function
def callback(in_data, frame_count, time_info, status):
    wf.writeframes(in_data)  # Write audio frames to WAV file
    return (in_data, pyaudio.paContinue)

# Open a stream for input only
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# Start recording
stream.start_stream()

# Keep recording for the specified duration
time.sleep(RECORD_SECONDS)

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Close the WAV file
wf.close()

print(f"Recording saved to {WAVE_OUTPUT_FILENAME}")
