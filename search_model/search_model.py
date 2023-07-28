import os
import io
import traceback
import pathlib
import librosa
from fastai.vision.all import *
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from flask import jsonify

warnings.filterwarnings("ignore")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceKey_GoogleCloud.json'

os_name = platform.system()

if os_name == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

BUCKET = "wav_audio_bits"
TOP_N = 20

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
F_MIN = 20
F_MAX = 8000
SAMPLE_RATE = 16000
AUDIO_LENGTH = 5
AUDIO_SAMPLES = int(AUDIO_LENGTH * SAMPLE_RATE)



@njit
def get_cs(array1, array2):

    dot_XY = np.dot(array1.reshape(1, -1), array2.reshape(-1, 1))[0][0]
    norm_X = np.sqrt(np.sum(array1**2))
    norm_Y = np.sqrt(np.sum(array2**2))
    cs = dot_XY / (norm_X * norm_Y)

    return cs

@njit
def get_all_cs(array1, array2):
    cs_array = np.zeros((array2.shape[0], 1))
    for i in range(array2.shape[0]):
        cs_array[i] = get_cs(array1, array2[i, :])
    return cs_array
####################################################################



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


#####################################################################
def getpath(file):
    return os.path.join(os.getcwd(),'search_model',file)

def audio_similarity_search(audio_path):
    try:
        # Convert the audio data to a numpy array
        clip, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
        file_duration = len(clip) / sample_rate

        # Load the necessary data and models if not already loaded
        data = pd.read_csv(getpath('resnet_feat.csv'))

        model1 = load_learner(getpath('resnet18_v12.pkl'))
        model2 = load_learner(getpath('resnet18_v20.pkl'))

        feat_scaler = pickle.load(open(getpath('feat_scaler.pkl'), 'rb'))
        feat_pca = pickle.load(open(getpath('feat_pca.pkl'), 'rb'))

        # Process the uploaded audio data and get the features
        get_mel_spec(
            clip=clip,
            sample_rate=sample_rate,
            max_size=AUDIO_SAMPLES,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX
        )

        if file_duration >= 1 and file_duration <= 5:
            uploaded_file_class = model1.predict('mel_spec.png')[0]
            uploaded_file_feat = get_features(model=model1, image_path="mel_spec.png")
        else:
            uploaded_file_class = model2.predict('mel_spec.png')[0]
            uploaded_file_feat = get_features(model=model2, image_path="mel_spec.png")

        uploaded_file_feat = feat_scaler.transform(uploaded_file_feat.reshape(1, -1))
        uploaded_file_feat = feat_pca.transform(uploaded_file_feat)

        class_df = data[data['class'] == uploaded_file_class]
        class_df.reset_index(drop=True, inplace=True)

        class_array = np.float32(class_df.drop(columns=["class", "file"]).values)

        cs_array = get_all_cs(array1=uploaded_file_feat.copy(), array2=class_array.copy())

        class_df = class_df.drop(columns=[f"feat{i + 1}" for i in range(1500)])

        class_df["similarity"] = cs_array
        class_df.sort_values("similarity", ascending=False, inplace=True)

        similar_file_names = list(class_df['file'].values)[:TOP_N]

        audio_list = []
        for file_name in similar_file_names:
            file_name = file_name.replace(" ", "")
            url_link = f"https://storage.cloud.google.com/{BUCKET}/{file_name.replace('#', '%23')}"
            audio_data = {
                "name": str(file_name),
                "link": str(url_link)
            }
            audio_list.append(audio_data)

        return audio_list
    except:
        print(traceback.print_exc())

# ad_path = r'D:\WebSearch\OF_WebSearch\audiofiles\pluck-loop-91bpm-132429.mp3'
# similar_file_names = audio_similarity_search(ad_path)
# print(similar_file_names)
