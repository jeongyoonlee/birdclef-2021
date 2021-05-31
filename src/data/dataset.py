import os

import pickle
import librosa
import soundfile
import numpy as np
from torch.utils.data import Dataset

#from data.transforms import *
from data.transforms import get_wav_transforms, crop_or_pad, mono_to_color, normalize
from params import MEAN, STD, CLASSES, OUTPUT_PATH


ONE_HOT = np.eye(len(CLASSES))
CONF_PATH = OUTPUT_PATH / "preds_oof.pkl"


def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale

    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max

    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


class BirdDataset(Dataset):
    """
    Torch dataset for the problem
    """

    def __init__(self, df, params, train=True, use_conf=False):
        """
        Constructor

        Arguments:
            df {pandas dataframe} -- Metadata
            params {AudioParams} -- Audio parameters

        Keyword Arguments:
            train {bool} -- Whether the dataset is used for training or validation (default: {True})
            use_conf {bool} -- Whether to use confidence for cropping (default: {False})
        """
        self.train = train
        self.params = params

        self.wav_transfos = get_wav_transforms() if train else None

        self.y = np.array([CLASSES.index(c) for c in df["primary_label"]])
        self.paths = df["file_path"].values

        self.sample_len = params.duration * params.sr

        self.use_conf = use_conf
        if use_conf:
            with open(CONF_PATH, "rb") as file:
                self.confidences = pickle.load(file)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y, sr = soundfile.read(self.paths[idx])

        if self.use_conf:
            path = "/".join(self.paths[idx].split('/')[-2:])
            confs = self.confidences[path][:, self.y[idx]]
            if len(confs):
                confs = confs / np.sum(confs)
            else:
                confs = None
        else:
            confs = None

        y = crop_or_pad(
            y, self.sample_len, sr=self.params.sr, train=self.train, probs=confs
        )

        if self.wav_transfos is not None:
            y = self.wav_transfos(y, self.params.sr)

        melspec = compute_melspec(y, self.params)

        image = mono_to_color(melspec)
        image = normalize(image, mean=None, std=None)

        return image, ONE_HOT[self.y[idx]]
