# """
# Stream from a microphone into SeamlessM4T ASR model indefinitely, convert audio data to tensor, and save audio files
# """
import pyaudio
import wave
import numpy as np
import torch

# Define the stream parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Mono audio
RATE = 16000              # Sample rate
CHUNK = 1000              # Number of frames per buffer
PROCESS_SAMPLES = 80000   # Number of samples to process at a time
save_audio = False  # Set to True if you want to save audio chunks to files

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

# Continuously record and process in chunks of 80000 samples
while True:
    # Check if we have more than 80000 samples
    if len(audio_data_chunks) * CHUNK >= PROCESS_SAMPLES:
        # Calculate the number of chunks to process
        num_chunks_to_process = PROCESS_SAMPLES // CHUNK
        # Convert the list of audio data chunks to a numpy array for the current segment
        audio_data_np = np.concatenate(audio_data_chunks[:num_chunks_to_process])
        
        if save_audio:
            wf, filename = open_new_file(0)  # Open a new file for the chunk if saving is enabled
            print(f"Recording to {filename}")
            # Write the numpy array data to the WAV file if saving is enabled
            wf.writeframes(audio_data_np.tobytes())
            # Close the current WAV file
            wf.close()
            print(f"Chunk saved to {filename}")

        # Convert the numpy array to a PyTorch tensor
        audio_data_tensor = torch.from_numpy(audio_data_np).float()

        # Remove processed chunks from the cache
        del audio_data_chunks[:num_chunks_to_process]

        # TODO: Add ASR inference here.


