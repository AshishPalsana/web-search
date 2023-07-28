import numpy as np
import librosa
from numba import njit
import numba.np.unsafe.ndarray
import matplotlib.pyplot as plt
from fastai.vision.all import *

@njit
def find_notnull_idx(x):
    x_idx = 0
    for i in range(len(x)):
        if x[i] != 0:
            x_idx = i
            break
        else:
            continue
    return x_idx

def get_mel_spec(clip, sample_rate, max_size, n_fft, hop_length, n_mels, fmin, fmax):

    clip_idx = find_notnull_idx(clip)
    clip = clip[clip_idx:]
    clip = clip[clip!=0]
    if len(clip) < max_size:
        n_repeat = int((max_size - (max_size % len(clip))) / len(clip) + 1)
        # clip = np.concatenate([clip for i in range(n_repeat)])
        clip = np.concatenate([clip, np.zeros((n_repeat-1)*len(clip))])
        del (n_repeat)
    clip = clip[:max_size]

    mel_spec = librosa.feature.melspectrogram(
        y=clip, sr=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        power=2.0, fmin=fmin, fmax=fmax
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.imsave("mel_spec.png", mel_spec_db)

inputs = None
def hook_fn(module, input, output):
    global inputs
    inputs = None
    inputs = input[0]

def get_features(model, image_path):

    # Attach the hook to the last layer of the model
    model.model[-1].register_forward_hook(hook_fn)

    # Load and preprocess your image
    image = PILImage.create(image_path)
    image = model.dls.after_item(image)
    image = model.dls.after_batch(image)

    # Pass the preprocessed image through the model to trigger the hook
    with torch.no_grad():
        model.model(image[0].unsqueeze(0))

    return inputs[0].numpy().flatten()