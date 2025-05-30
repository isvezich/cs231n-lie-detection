import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio


class AudioCompressedVisualDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, jnp_dir, num_tokens=64, frame_size=32):
        super(AudioCompressedVisualDataset, self).__init__()

        self.annos = pd.read_csv(annotations_file)  # [file_name, label, gender]
        self.audio_dir = audio_dir  # all files in '.wav' format
        self.num_tokens = num_tokens

        self.jnp_dir = jnp_dir
        self.frame_size = frame_size  # 224 or any multiple of 32
        self.transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            # T.Resize(frame_size),
            # normalize to imagenet mean and std values
            T.Normalize(mean=[0.456], std=[0.225]),
        ])
        self.number_of_target_frames = num_tokens

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        # select one clip
        clip_name = self.annos.iloc[idx, 0]

        # get audio path
        # audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
        # # load the audio file with torch audio
        # waveform, sample_rate = torchaudio.load(audio_path)
        # # use mono audio instead os stereo audio (use left by default)
        # waveform = waveform[0]
        #
        # # calculate duration of the audio clip
        # clip_duration = len(waveform) / sample_rate
        # """
        # # for wav2vec2, 1 Token corresponds to ~ 321.89 discrete samples
        # # to get precisely 64 tokens (a hyperparameter that can be changed), the length of input discrete samples to the model should be 321.89 * 64
        # # divide the above by the clip duration to get new sample rate (or) new_sample_rate * clip_duration = 321.89 * num tokens
        # """
        # new_sample_rate = int(321.893491124260 * self.num_tokens / clip_duration)
        # # resample
        # waveform = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
        # # required by collate function
        # mono_waveform = waveform.unsqueeze(0)
        # mono_waveform.type(torch.float32)
        mono_waveform = None

        #  get compressed face features path
        file_path = self.jnp_dir + clip_name + '.npy'

        frames = np.load(file_path)

        # print("LOADLED FRAMES")
        # print(frames.shape)

        frames = frames[0, :, :, :, :]
        # print("AFTER SQ")
        # print(frames.shape)
        if frames.shape[0] > self.number_of_target_frames:
            index = np.random.choice(frames.shape[0], self.number_of_target_frames, replace=False)
            frames = frames[index]
        # print("FRAMES AFTER")
        # print(frames.shape)

        # frame_names = [i for i in os.listdir(file_path) if i.split('.')[-1] == 'npy']

        # sample 64 face frames
        target_frames = np.linspace(0, frames.shape[0] - 1, num=self.number_of_target_frames)
        target_frames = np.around(target_frames).astype(
            int)  # certain frames may be redundant because of rounding to the nearest integer
        face_frames = []
        for i in target_frames:
            single_frame = frames[i, :, :, :]
            face_frames.append(self.transforms(single_frame))

        # print("FACE FRAME IS")
        # print(len(face_frames))
        if len(face_frames) < 64:
            raise Exception(f"FOUND FACE FRAME SIZE {file_path} {len(face_frames)}")
        face_frames = torch.stack(face_frames, 0)
        face_frames.type(torch.float32)
        # print("FRAMES FINAL")
        # print(face_frames.shape)
        face_frames.type(torch.float32)

        # assign integer to labels
        str_label = self.annos.iloc[idx, 1]
        str_label = str_label.replace(" ", "")
        if str_label == 'truth' or str_label == 'Truth' or str_label == 'truthful' or str_label == 0:
            label = 0
        elif str_label == 'deception' or str_label == 'Deception' or str_label == 'lie' or str_label == 1:
            label = 1
        # undefined label error
        else:
            print("LABEL WAS")
            print(str_label)
            raise Exception(f"undefined label isss {str_label}")

        return mono_waveform, face_frames, label


def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def af_collate_fn_compressed(batch):
    tensors, face_tensors, targets = [], [], []

    # Gather in lists, and encode labels as indices
    for waveform, face_frames, label in batch:
        tensors += [waveform]
        face_tensors += [face_frames]
        targets += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    # tensors = af_pad_sequence(tensors)
    tensors = []
    face_tensors = torch.stack(face_tensors)
    targets = torch.stack(targets)

    return tensors, face_tensors, targets
