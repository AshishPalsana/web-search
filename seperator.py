import multiprocessing
import os
from spleeter.separator import Separator

if __name__ == '__main__':
    multiprocessing.freeze_support()

    config = 'spleeter:4stems'

    # Create an instance of the spleeter separator
    separator = Separator(config)

    audio_file = 'audiofiles/CVUUZH9-drum-bass-orchestra.mp3'

    # Define the output directory for the separated components
    output_dir = 'audiofiles'
    # Separate the audio into its components
    separator.separate_to_file(audio_file, output_dir)

    # Get the paths to the separated audio files
    drums_path = os.path.join(output_dir, 'drums.wav')
    synth_path = os.path.join(output_dir, 'other.wav')
    vocals_path = os.path.join(output_dir, 'vocals.wav')
    bass_path = os.path.join(output_dir, 'bass.wav')

    print(separator)