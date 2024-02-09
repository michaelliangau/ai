# """
# Stream from a microphone into SeamlessM4T ASR model indefinitely, convert audio data to tensor, and save audio files
# """
import pyaudio
import wave
import itertools
import time
import numpy as np
import torch

# Define the stream parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate
CHUNK = 1000              # Number of frames per buffer
RECORD_SECONDS = 5        # Duration of recording chunk in seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

def open_new_file(index):
    # Open a new WAV file for writing
    wave_output_filename = f"tmp_{index}.wav"  # Temporary output WAV file
    wf = wave.open(wave_output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    return wf, wave_output_filename

# Initialize an empty list to store chunks of audio data
audio_data_chunks = []

# Define the callback function
def callback(in_data, frame_count, time_info, status):
    audio_data_chunks.append(np.frombuffer(in_data, dtype=np.int16))  # Append the chunk of audio data
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
save_audio = True  # Set to True if you want to save audio chunks to files

# Record indefinitely and process in chunks
for i in itertools.count():
    # Reset the audio_data_chunks list for the new recording chunk
    audio_data_chunks = []
    
    # Record for exactly 5 seconds
    frames_needed = RATE * RECORD_SECONDS
    while len(audio_data_chunks) * CHUNK < frames_needed:
        time.sleep(0.1)  # Sleep briefly to avoid busy waiting

    # After recording, truncate the list of audio data chunks to get exactly 44100 * 5 samples
    samples_collected = len(audio_data_chunks) * CHUNK
    if samples_collected > frames_needed:
        excess_samples = samples_collected - frames_needed
        last_chunk = audio_data_chunks[-1]
        audio_data_chunks[-1] = last_chunk[:-excess_samples]

    # Convert the list of audio data chunks to a numpy array
    audio_data_np = np.concatenate(audio_data_chunks)
    
    if save_audio:
        wf, filename = open_new_file(i)  # Open a new file for each chunk if saving is enabled
        print(f"Recording to {filename}")
        # Write the numpy array data to the WAV file if saving is enabled
        wf.writeframes(audio_data_np.tobytes())
        # Close the current WAV file
        wf.close()
        print(f"Chunk {i} saved to {filename}")

    # Convert the numpy array to a PyTorch tensor
    audio_data_tensor = torch.from_numpy(audio_data_np).float()

    # Here you can process the tensor with your ASR model
    print(f"Chunk {i} processed and converted to tensor with shape {audio_data_tensor.shape}")

