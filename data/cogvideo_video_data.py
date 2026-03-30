"""
Dataset helpers for CogVideoX-backed VideoDPO training.
The format is intentionally compatible with the original VideoDPO metadata/pair.json layout,
but resolves relative paths more robustly for external config folders.
"""

import json
import random
from pathlib import Path

import torch
import yaml
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class TextVideo(Dataset):
    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
    ):
        self.data_root = data_root
        self.data_root_path = Path(data_root).expanduser().resolve()
        self.data_root_dir = self.data_root_path.parent
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.clip_length = clip_length
        assert self.subset_split in ["train", "test", "all"]

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]

        self._make_dataset()

    def _resolve_path(self, path_like):
        path = Path(path_like).expanduser()
        if path.is_absolute():
            return path

        candidates = [
            self.data_root_dir / path,
            Path.cwd() / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return (self.data_root_dir / path).resolve()

    def _make_dataset(self):
        with open(self.data_root_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        self.videos = []
        for meta_path in self.config["META"]:
            meta_path = self._resolve_path(meta_path)
            metadata_path = meta_path / "metadata.json"
            with open(metadata_path, "r") as f:
                videos = json.load(f)
                for item in videos:
                    if item["basic"]["clip_duration"] < self.clip_length:
                        continue
                    item["basic"]["clip_path"] = str(
                        (meta_path / item["basic"]["clip_path"]).resolve()
                    )
                    self.videos.append(item)
        print(f"Number of videos = {len(self.videos)}")

    def __getitem__(self, index):
        while True:
            video_path = Path(self.videos[index]["basic"]["clip_path"])
            try:
                video_reader = VideoReader(
                    str(video_path),
                    ctx=cpu(0),
                    width=self.resolution[1],
                    height=self.resolution[0],
                )
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                break
            except Exception:
                index += 1
                print(f"Load video failed! path = {video_path}")
                return self.__getitem__(index)

        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))

        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = all_frames[rand_idx : rand_idx + self.video_length]
        frames = video_reader.get_batch(frame_indices)
        assert (
            frames.shape[0] == self.video_length
        ), f"{len(frames)}, self.video_length={self.video_length}"

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        assert (
            frames.shape[2] == self.resolution[0]
            and frames.shape[3] == self.resolution[1]
        ), f"frames={frames.shape}, self.resolution={self.resolution}"
        frames = (frames / 255 - 0.5) * 2
        return {
            "video": frames,
            "caption": self.videos[index]["misc"]["frame_caption"][0],
        }

    def __len__(self):
        return len(self.videos)


class TextVideoDPO(Dataset):
    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
        dupbeta=1.0,
    ):
        self.data = TextVideo(
            data_root, resolution, video_length, frame_stride, subset_split, clip_length
        )
        self.pairs = []
        self.data_root = data_root
        self.dupbeta = dupbeta
        with open(self.data.data_root_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        for meta_path in self.config["META"]:
            meta_path = self.data._resolve_path(meta_path)
            pairdata_path = meta_path / "pair.json"
            with open(pairdata_path, "r") as f:
                pairs = json.load(f)
                for item in pairs:
                    if dupbeta:
                        score = item.get("score", 1)
                        sample = [item["video1"], item["video2"], item["frame_caption"], score]
                    else:
                        sample = [item["video1"], item["video2"], item["frame_caption"]]
                    self.pairs.append(sample)
        print(f"DPO dataset has {self.__len__()} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if self.dupbeta:
            videowidx, videolidx, frame_caption, prob_score = self.pairs[index]
            dupfactor = (0.72 / prob_score) ** self.dupbeta
        else:
            videowidx, videolidx, frame_caption = self.pairs[index]
            dupfactor = 1.0

        videow = self.data[videowidx]["video"]
        videol = self.data[videolidx]["video"]
        combined_frames = torch.cat([videow, videol], dim=0)
        if isinstance(frame_caption, list):
            frame_caption = frame_caption[0]

        return {"video": combined_frames, "caption": frame_caption, "dupfactor": dupfactor}
