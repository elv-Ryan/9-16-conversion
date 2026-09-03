import os
import cv2
import numpy as np
import torch
import torch.nn as nn
# from torchvision.models.inception import Inception3
from .inception import Inception3

class _ToTensorNormalize:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, images):
        """(N), H, W, C  ->  (N), C, H, W"""
        if len(images.shape) == 4:
            images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to(torch.float32) / 255
        else:
            images = torch.from_numpy(images.copy()).permute(2, 0, 1).to(torch.float32) / 255
        images -= self.mean[:, None, None]
        images /= self.std[:, None, None]
        return images


class _Identity(nn.Module):

    def forward(self, x):
        return x


class _FeatureExtractor(Inception3):

    def __init__(self):
        super().__init__()
        self.fc = _Identity()

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        return x


class TestCardClassifier:

    def __init__(self, device, model_dir, n_frames):
        # normalized test card features
        path_to_features = os.path.join(model_dir, "test_card_features.npy")
        self.test_card_features = torch.from_numpy(np.load(path_to_features))
        self.device = device
        # feature extractor
        self.feature_extractor = _FeatureExtractor()
        path_to_model = os.path.join(model_dir, "feature_extractor.pth")
        self.feature_extractor.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

        # transform
        self.transform = _ToTensorNormalize()

        self.n_frames = n_frames
        self.width = 299
        self.height = 299

    def _sample_frames(self, path_to_video, shots):
        # sample evenly spaced frames from each shot
        step = (shots[:, 1] - shots[:, 0] + 1) // self.n_frames
        frame_inds = np.repeat(shots[:, 0, np.newaxis], self.n_frames, axis=1)
        frame_inds += (step // 2)[:, None]
        frame_inds += np.arange(self.n_frames, dtype=np.int32)[None, :] * step[:, None]

        # frame index list and range for each shot in the sampled frames
        frame_list = set()
        shot_ranges = np.zeros_like(shots)
        start_index = 0
        for i in range(shots.shape[0]):
            inds = set(frame_inds[i])
            n_frames = len(inds)
            shot_ranges[i] = start_index, start_index + n_frames
            frame_list |= inds
            start_index += n_frames

        # read frames
        cap = cv2.VideoCapture(path_to_video)
        frames = []
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_index in frame_list:
                    frame = cv2.resize(frame, (self.width, self.height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                frame_index += 1
            else:
                break
        frames = np.stack(frames, axis=0)
        cap.release()

        return shot_ranges, frames

    def predict(self, path_to_video, shots, mask):
        scores = torch.zeros(mask.shape, dtype=torch.float32)

        # empty mask
        if mask.sum() == 0:
            return scores

        # sample frames for each shot in the mask
        shots_, frames = self._sample_frames(path_to_video, shots[mask])

        # extract features
        frames = self.transform(frames).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(frames)
        features = features.cpu()

        # cosine similarities
        features = features / features.norm(dim=1)[:, None]
        sim = torch.mm(self.test_card_features, features.transpose(0, 1))
        sim, _ = sim.max(dim=0)
        sim = torch.clamp((sim + 1) / 2, 0, 1)

        # average scores for each shot
        scores[mask] = torch.tensor([sim[range(*shot_)].mean() for shot_ in shots_], dtype=scores.dtype)
        return scores


