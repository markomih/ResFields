import numpy as np
import os.path as osp
import cv2
import torch
from typing import Dict, Optional, Tuple
from . import io
from . import geometry
from utils.misc import get_rank
import sys
# sys.path.append("lib")
# from utils_torch import common
import logging

class MetadataInfoMixin:
    @property
    def frame_names(self):
        raise NotImplementedError

    @property
    def time_ids(self):
        raise NotImplementedError

    @property
    def camera_ids(self):
        raise NotImplementedError

    @property
    def uniq_time_ids(self):
        return np.unique(self.time_ids)

    @property
    def uniq_camera_ids(self):
        return np.unique(self.camera_ids)

    @property
    def num_frames(self):
        return len(self.frame_names)

    @property
    def num_times(self):
        return len(set(self.time_ids))

    @property
    def num_cameras(self):
        return len(set(self.camera_ids))

    @property
    def embeddings_dict(self):
        return {"time": self.uniq_time_ids, "camera": self.uniq_camera_ids}

    def validate_metadata_info(self):
        if not (np.ediff1d(self.uniq_time_ids) == 1).all():
            raise ValueError("Unique time ids are not consecutive.")
        if not (np.ediff1d(self.uniq_camera_ids) == 1).all():
            raise ValueError("Unique camera ids are not consecutive.")


class Parser(MetadataInfoMixin):
    """Parser for parsing and loading raw data without any preprocessing or
    data splitting.
    """

    def __init__(
        self,
        dataset: str,
        sequence: str,
        data_root: str,
    ):
        self.dataset = dataset
        self.sequence = sequence
        self.data_root = data_root or osp.abspath(
            osp.join(osp.dirname(__file__), "..", "..", "datasets")
        )
        self.data_dir = osp.join(self.data_root, self.dataset, self.sequence)

    def load_rgba(self, time_id: int, camera_id: int) -> np.ndarray:
        raise NotImplementedError(
            f"Load RGBA with time_id={time_id}, camera_id={camera_id}."
        )

    def load_depth(self, time_id: int, camera_id: int) -> np.ndarray:
        raise NotImplementedError(
            f"Load depth with time_id={time_id}, camera_id={camera_id}."
        )

    def load_camera(
        self, time_id: int, camera_id: int, **_
    ):
        raise NotImplementedError(
            f"Load camera with time_id={time_id}, camera_id={camera_id}."
        )

    def load_covisible(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Load covisible with time_id={time_id}, "
            f"camera_id={camera_id}, split={split}."
        )

    def load_dynamic_pixel(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Load dynamic_pixel with time_id={time_id}, "
            f"camera_id={camera_id}, split={split}."
        )

    def load_keypoints(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Load keypoints with time_id={time_id}, "
            f"camera_id={camera_id}, split={split}."
        )

    def load_skeleton(self, split: str):
        raise NotImplementedError(f"Load skeleton with split={split}.")

    def load_split(
        self, split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(f"Load split {split}.")

    @property
    def center(self):
        raise NotImplementedError

    @property
    def scale(self):
        raise NotImplementedError

    @property
    def near(self):
        raise NotImplementedError

    @property
    def far(self):
        raise NotImplementedError

    @property
    def factor(self):
        raise NotImplementedError

    @property
    def fps(self):
        raise NotImplementedError

    @property
    def bbox(self):
        raise NotImplementedError

    @property
    def lookat(self):
        raise NotImplementedError

    @property
    def up(self):
        raise NotImplementedError


def _load_scene_info(
    data_dir: str,
) -> Tuple[np.ndarray, float, float, float]:
    scene_dict = io.load(osp.join(data_dir, "scene.json"))
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]
    return center, scale, near, far


def _load_metadata_info(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dict = io.load(osp.join(data_dir, "dataset.json"))
    _frame_names = np.array(dataset_dict["ids"])

    metadata_dict = io.load(osp.join(data_dir, "metadata.json"))
    time_ids = np.array(
        [metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32
    )
    camera_ids = np.array(
        [metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32
    )

    frame_names_map = np.zeros(
        (time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype
    )
    for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
        frame_names_map[t, c] = _frame_names[i]

    return frame_names_map, time_ids, camera_ids


class NerfiesParser(Parser):
    """Parser for the Nerfies dataset."""

    SPLITS = [
        "train_intl",
        "train_mono",
        "val_intl",
        "val_mono",
        "train_common",
        "val_common",
    ]
    DEFAULT_FACTORS: Dict[str, int] = {
        "nerfies/broom": 4,
        "nerfies/curls": 8,
        "nerfies/tail": 4,
        "nerfies/toby-sit": 4,
        "hypernerf/3dprinter": 4,
        "hypernerf/chicken": 4,
        "hypernerf/peel-banana": 4,
    }
    DEFAULT_FPS: Dict[str, int] = {
        "nerfies/broom": 15,
        "nerfies/curls": 5,
        "nerfies/tail": 15,
        "nerfies/toby-sit": 15,
        "hypernerf/3dprinter": 15,
        "hypernerf/chicken": 15,
        "hypernerf/peel-banana": 15,
    }
    def __init__(
        self,
        dataset: str,
        sequence: str,
        data_root: str,
        factor: Optional[int] = None,
        fps: Optional[float] = None,
        use_undistort: bool = False,
    ):
        super().__init__(dataset, sequence, data_root=data_root)

        self._factor = factor or self.DEFAULT_FACTORS[f"{dataset}/{sequence}"]
        self._fps = fps or self.DEFAULT_FPS[f"{dataset}/{sequence}"]
        self.use_undistort = use_undistort

        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = _load_scene_info(self.data_dir)
        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = _load_metadata_info(self.data_dir)
        self._load_extra_info()

        self.splits_dir = osp.join(self.data_dir, "splits")
        if not osp.exists(self.splits_dir):
            self._create_splits()

    def load_rgba(
        self,
        time_id: int,
        camera_id: int,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        rgb_path = osp.join(
            self.data_dir,
            "rgb" if not use_undistort else "rgb_undistort",
            f"{self._factor}x",
            frame_name + ".png",
        )
        if osp.exists(rgb_path):
            rgba = io.load(rgb_path, flags=cv2.IMREAD_UNCHANGED)
            if rgba.shape[-1] == 3:
                rgba = np.concatenate(
                    [rgba, np.full_like(rgba[..., :1], 255)], axis=-1
                )
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            rgb = self.load_rgba(time_id, camera_id, use_undistort=False)[
                ..., :3
            ]
            rgb = cv2.undistort(rgb, camera.intrin, camera.distortion)
            alpha = (
                cv2.undistort(
                    np.full_like(rgb, 255),
                    camera.intrin,
                    camera.distortion,
                )
                == 255
            )[..., :1].astype(np.uint8) * 255
            rgba = np.concatenate([rgb, alpha], axis=-1)
            io.dump(rgb_path, rgba)
        else:
            raise ValueError(f"RGB image not found: {rgb_path}.")
        return rgba

    def load_camera(
        self,
        time_id: int,
        camera_id: int,
        use_undistort: Optional[bool] = None,
    ) -> geometry.Camera:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        camera = (
            geometry.Camera.fromjson(
                osp.join(self.data_dir, "camera", frame_name + ".json")
            )
            .rescale_image_domain(1 / self._factor)
            .translate(-self._center)
            .rescale(self._scale)
        )
        if use_undistort:
            camera = camera.undistort_image_domain()
        return camera

    def load_covisible(
        self,
        time_id: int,
        camera_id: int,
        split: str,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        covisible_path = osp.join(
            self.data_dir,
            "covisible" if not use_undistort else "covisible_undistort",
            f"{self._factor}x",
            split,
            frame_name + ".png",
        )
        if osp.exists(covisible_path):
            # (H, W, 1) uint8 mask.
            covisible = io.load(covisible_path)[..., :1]
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            covisible = self.load_covisible(
                time_id,
                camera_id,
                split,
                use_undistort=False,
            ).repeat(3, axis=-1)
            alpha = (
                cv2.undistort(
                    np.full_like(covisible, 255),
                    camera.intrin,
                    camera.distortion,
                )
                == 255
            )[..., :1].astype(np.uint8) * 255
            covisible = cv2.undistort(
                covisible, camera.intrin, camera.distortion
            )[..., :1]
            covisible = ((covisible == 255) & (alpha == 255)).astype(
                np.uint8
            ) * 255
            io.dump(covisible_path, covisible)
        else:
            raise ValueError(
                f"Covisible image not found: {covisible_path}. If not "
                f"processed before, please consider running "
                f"tools/process_covisible.py."
            )
        return covisible

    def load_dynamic_pixel(
        self,
        time_id: int,
        camera_id: int,
        split: str,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        # DYNAMIC TODO: add dynamic pixels.
        # raise NotImplementedError("Dynamic pixels not implemented yet.")
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        dynamic_pixel_path = osp.join(
            self.data_dir,
            "dynamic_pixel" if not use_undistort else "dynamic_pixel_undistort",
            f"{self._factor}x",
            split,
            frame_name + ".png",
        )
        if osp.exists(dynamic_pixel_path):
            # (H, W, 1) uint8 mask.
            dynamic_pixel = io.load(dynamic_pixel_path)[..., :1]
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            dynamic_pixel = self.load_dynamic_pixel(
                time_id,
                camera_id,
                split,
                use_undistort=False,
            ).repeat(3, axis=-1)
            alpha = (
                cv2.undistort(
                    np.full_like(dynamic_pixel, 255),
                    camera.intrin,
                    camera.distortion,
                )
                == 255
            )[..., :1].astype(np.uint8) * 255
            dynamic_pixel = cv2.undistort(
                dynamic_pixel, camera.intrin, camera.distortion
            )[..., :1]
            dynamic_pixel = ((dynamic_pixel == 255) & (alpha == 255)).astype(
                np.uint8
            ) * 255
            io.dump(dynamic_pixel_path, dynamic_pixel)
        else:
            raise ValueError(
                f"dynamic_pixel image not found: {dynamic_pixel_path}. If not "
                f"processed before, please consider running "
                f"tools/process_dynamic_pixel.py."
                f"Though it probably doesn't exist yet."
            )
        return dynamic_pixel
    def load_keypoints(
        self,
        time_id: int,
        camera_id: int,
        split: str,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        keypoints_path = osp.join(
            self.data_dir,
            "keypoint" if not use_undistort else "keypoint_undistort",
            f"{self._factor}x",
            split,
            frame_name + ".json",
        )
        if osp.exists(keypoints_path):
            camera = self.load_camera(
                time_id, camera_id, use_undistort=use_undistort
            )
            offset = 0.5 if camera.use_center else 0
            # (J, 3).
            keypoints = np.array(io.load(keypoints_path), np.float32)
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            offset = 0.5 if camera.use_center else 0
            keypoints = self.load_keypoints(
                time_id,
                camera_id,
                split,
                use_undistort=False,
            )
            keypoints = np.concatenate(
                [
                    camera.undistort_pixels(keypoints[:, :2]) - offset,
                    keypoints[:, -1:],
                ],
                axis=-1,
            )
            keypoints[keypoints[:, -1] == 0] = 0
            io.dump(keypoints_path, keypoints)
        else:
            raise ValueError(
                f"Keypoints not found: {keypoints_path}. If not "
                f"annotated before, please consider running "
                f"tools/annotate_keypoints.ipynb."
            )
        return np.concatenate(
            [keypoints[:, :2] + offset, keypoints[:, -1:]], axis=-1
        )

    def load_skeleton(
        self,
        split: str,
        use_undistort: Optional[bool] = None,
    ):
        if use_undistort is None:
            use_undistort = self.use_undistort

        skeleton_path = osp.join(
            self.data_dir,
            "keypoint",
            f"{self._factor}x",
            split,
            "skeleton.json",
        )
        if osp.exists(skeleton_path):
            skeleton = visuals.Skeleton(
                **{
                    k: v
                    for k, v in io.load(skeleton_path).items()
                    if k != "name"
                }
            )
        elif use_undistort:
            skeleton = self.load_skeleton(split, use_undistort=False)
            io.dump(skeleton_path, skeleton)
        else:
            raise ValueError(
                f"Skeleton not found: {skeleton_path}. If not "
                f"annotated before, please consider running "
                f"tools/annotate_keypoints.ipynb."
            )
        return skeleton

    def load_split(
        self, split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert split in self.SPLITS

        split_dict = io.load(osp.join(self.splits_dir, f"{split}.json"))
        return (
            np.array(split_dict["frame_names"]),
            np.array(split_dict["time_ids"], np.uint32),
            np.array(split_dict["camera_ids"], np.uint32),
        )

    def load_bkgd_points(self) -> np.ndarray:
        bkgd_points = io.load(osp.join(self.data_dir, "points.npy")).astype(
            np.float32
        )
        bkgd_points = (bkgd_points - self._center) * self._scale
        return bkgd_points

    def _load_extra_info(self) -> None:
        extra_path = osp.join(self.data_dir, "extra.json")
        if osp.exists(extra_path):
            extra_dict = io.load(extra_path)
            bbox = np.array(extra_dict["bbox"], dtype=np.float32)
            lookat = np.array(extra_dict["lookat"], dtype=np.float32)
            up = np.array(extra_dict["up"], dtype=np.float32)
        else:
            cameras = common.parallel_map(
                self.load_camera, self._time_ids, self._camera_ids
            )
            bkgd_points = self.load_bkgd_points()
            points = np.concatenate(
                [
                    bkgd_points,
                    np.array([c.position for c in cameras], np.float32),
                ],
                axis=0,
            )
            bbox = np.stack([points.min(axis=0), points.max(axis=0)])
            lookat = geometry.utils.tringulate_rays(
                np.stack([c.position for c in cameras], axis=0),
                np.stack([c.optical_axis for c in cameras], axis=0),
            )
            up = np.mean([c.up_axis for c in cameras], axis=0)
            up /= np.linalg.norm(up)
            extra_dict = {
                "factor": self._factor,
                "fps": self._fps,
                "bbox": bbox.tolist(),
                "lookat": lookat.tolist(),
                "up": up.tolist(),
            }
            logging.info(
                f'Extra info not found. Dumping extra info to "{extra_path}."'
            )
            io.dump(extra_path, extra_dict)

        self._bbox = bbox
        self._lookat = lookat
        self._up = up

    @property
    def frame_names(self):
        return self._frame_names_map[self.time_ids, self.camera_ids]

    @property
    def time_ids(self):
        return self._time_ids

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def center(self):
        return self._center

    @property
    def scale(self):
        return self._scale

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def factor(self):
        return self._factor

    @property
    def fps(self):
        return self._fps

    @property
    def bbox(self):
        return self._bbox

    @property
    def lookat(self):
        return self._lookat

    @property
    def up(self):
        return self._up


class iPhoneParser(NerfiesParser):
    """Parser for the Nerfies dataset."""

    SPLITS = [
        "train",
        "val",
    ]

    def __init__(
        self,
        dataset: str,
        sequence: str,
        data_root: str,
    ):
        super(NerfiesParser, self).__init__(
            dataset, sequence, data_root=data_root
        )
        self.use_undistort = False

        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = _load_scene_info(self.data_dir)
        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = _load_metadata_info(self.data_dir)
        self._load_extra_info()

        self.splits_dir = osp.join(self.data_dir, "splits")
        if not osp.exists(self.splits_dir):
            self._create_splits()

    def load_rgba(self, time_id: int, camera_id: int) -> np.ndarray:
        return super().load_rgba(time_id, camera_id, use_undistort=False)

    def load_depth(
        self,
        time_id: int,
        camera_id: int,
    ) -> np.ndarray:
        frame_name = self._frame_names_map[time_id, camera_id]
        depth_path = osp.join(
            self.data_dir, "depth", f"{self._factor}x", frame_name + ".npy"
        )
        depth = io.load(depth_path) * self.scale
        camera = self.load_camera(time_id, camera_id)
        # The original depth data is projective; convert it to ray traveling
        # distance.
        depth = depth / camera.pixels_to_cosa(camera.get_pixels())
        return depth

    def load_camera(
        self, time_id: int, camera_id: int, **_
    ):
        return super().load_camera(time_id, camera_id, use_undistort=False)

    def load_covisible(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        return super().load_covisible(
            time_id, camera_id, split, use_undistort=False
        )

    def load_dynamic_pixel(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        return super().load_dynamic_pixel(
            time_id, camera_id, split, use_undistort=False
        )

    def load_keypoints(
        self, time_id: int, camera_id: int, split: str, **_
    ) -> np.ndarray:
        return super().load_keypoints(
            time_id, camera_id, split, use_undistort=False
        )

    def _load_extra_info(self) -> None:
        extra_path = osp.join(self.data_dir, "extra.json")
        extra_dict = io.load(extra_path)
        self._factor = extra_dict["factor"]
        # self._factor = DEFAULT_FACTOR
        self._fps = extra_dict["fps"]
        self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
        self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
        self._up = np.array(extra_dict["up"], dtype=np.float32)


class Dataset(MetadataInfoMixin):
    """Cached dataset for training and evaluation."""

    __parser_cls__: Parser = None

    def __init__(
        self,
        dataset: str,
        sequence: str,
        data_root: str,
        split: Optional[str] = None,
        training: Optional[bool] = None,
        cache_num_repeat: int = 1,
        seed: int = 0,
        same_time_sampling: bool = False,
        **kwargs,
    ):
        # super().__init__(daemon=True)
        super().__init__()
        self.rank = get_rank()

        self._num_repeats = 0
        self._index = 0

        self.dataset = dataset
        self.sequence = sequence

        self.split = split
        if training is None:
            training = self.split is not None and self.split.startswith(
                "train"
            )
        self.training = training
        self.cache_num_repeat = cache_num_repeat
        self.seed = seed
        self.same_time_sampling = same_time_sampling

        # RandomState's choice method is too slow.
        self.rng = np.random.default_rng(seed)

        assert self.__parser_cls__, "Parser class is not defined."
        self.parser = self.__parser_cls__(
            dataset=self.dataset,
            sequence=self.sequence,
            data_root=data_root
        )

    @classmethod
    def create(cls, *args, **kwargs):
        """A wrapper around __init__ such that always start fetching *after*
        subclasses get initialized.

        Note that __post_init__ does not work in this case.
        """
        dataset = cls(*args, **kwargs)
        # dataset.start()
        return dataset

    @classmethod
    def create_dummy(cls, *args, **kwargs):
        """Create dummy dataset such that no prefetching is performed.

        This method can be useful when evaluating.
        """
        dataset = cls(*args, **kwargs)
        return dataset

    @property
    def data_dir(self):
        return self.parser.data_dir

    @property
    def has_novel_view(self):
        raise NotImplementedError

    @property
    def has_keypoints(self):
        raise NotImplementedError

    @property
    def center(self):
        return self.parser.center

    @property
    def scale(self):
        return self.parser.scale

    @property
    def near(self):
        return self.parser.near

    @property
    def far(self):
        return self.parser.far

    @property
    def factor(self):
        return self.parser.factor

    @property
    def fps(self):
        return self.parser.fps

    @property
    def bbox(self):
        return self.parser.bbox

    @property
    def lookat(self):
        return self.parser.lookat

    @property
    def up(self):
        return self.parser.up


class NerfiesDataset(Dataset, MetadataInfoMixin):
    """Nerfies dataset for both Nerfies and HyperNeRF sequences.

    The following previous works are tested on this dataset:

    [1] Nerfies: Deformable Neural Radiance Fields.
        Park et al., ICCV 2021.
        https://arxiv.org/abs/2011.12948

    [2] HyperNeRF: A Higher-Dimensional Representation for Topologically
    Varying Neural Radiance Fields.
        Park et al., SIGGRAPH Asia 2021.
        https://arxiv.org/abs/2106.13228
    """

    __parser_cls__: Parser = NerfiesParser

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Filtered by split.
        self._frame_names, self._time_ids, self._camera_ids = self.parser.load_split(self.split)
        filter_time = kwargs.get('filter_time', None)
        if filter_time is not None:
            new_frame_names, new_time_ids, new_camera_ids = [], [], []
            for _fname, _tid, _cam_id in zip(self._frame_names, self._time_ids, self._camera_ids):
                if _tid < filter_time:
                    new_frame_names.append(_fname)
                    new_time_ids.append(_tid)
                    new_camera_ids.append(_cam_id)
            self._frame_names = np.array(new_frame_names).astype(self._frame_names.dtype)
            self._time_ids = np.array(new_time_ids).astype(self._time_ids.dtype)
            self._camera_ids = np.array(new_camera_ids).astype(self._camera_ids.dtype)

        self.cameras = [self.parser.load_camera(_t, _c) for _t, _c in zip(self._time_ids, self._camera_ids)]
        self.time_ids_torch = torch.from_numpy(self._time_ids.astype(np.int64)).to(device=self.rank)

        if self.training:
            self.validate_metadata_info()

        bkgd_points = self.parser.load_bkgd_points()
        self.bkgd_points = torch.from_numpy(bkgd_points).to(self.rank)

    @property
    def has_novel_view(self):
        return True

    @property
    def has_keypoints(self):
        return osp.exists(osp.join(self.data_dir, "keypoint"))

    @property
    def frame_names(self):
        return self._frame_names

    @property
    def time_ids(self):
        return self._time_ids

    @property
    def camera_ids(self):
        return self._camera_ids


class NerfiesDatasetFromAllFrames(NerfiesDataset):
    """Nerfies dataset for both Nerfies and HyperNeRF sequences."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # if self.training:
        rgbas = torch.from_numpy(np.array([
            self.parser.load_rgba(_t, _c)
            for _t, _c in zip(self._time_ids, self._camera_ids)
        ]))
        images, masks = rgbas[..., :3]/255., rgbas[..., -1:]

        # load validation masks
        self.covisible_masks, self.dynamic_pixel_masks = None, None
        if self.split != 'train':
            try:
                # Covisible is not necessary for evaluation.
                self.covisible_masks = torch.from_numpy(np.array([
                    self.parser.load_covisible(_t, _c, self.split)
                    for _t, _c in zip(self._time_ids, self._camera_ids)
                ])).float().to(self.rank)
            except ValueError:
                pass

        try:
            self.dynamic_pixel_masks = torch.from_numpy(np.array([
                self.parser.load_dynamic_pixel(_t, _c, self.split)
                for _t, _c in zip(self._time_ids, self._camera_ids)
            ])).float().to(self.rank)
        except ValueError:
            pass

        # move data to device
        # monocular video (same intrinsics)
        self.directions = torch.from_numpy(self.cameras[0].pixels_to_local_viewdirs(self.cameras[0].get_pixels())).to(self.rank) # torch.Size([H, W, 3])
        # self.directions = torch.from_numpy(np.array([c.pixels_to_local_viewdirs(_pix) for c in self.cameras])).to(self.rank)
        self.all_c2w = torch.inverse(torch.from_numpy(np.array([c.extrin for c in self.cameras])))[:, :3, :4].to(self.rank) #self.pose_all.float().to(self.rank)[:, :3, :4] # torch.Size([46, 3, 4])
        # self.all_intrinsics = torch.from_numpy(np.array([c.intrin for c in self.cameras])) 
        self.all_fg_masks = (masks > 0).to(self.rank).float().squeeze(-1) # torch.Size([46, 400, 400])
        self.all_images = images.float().to(self.rank)*self.all_fg_masks[..., None].float() # torch.Size([46, 400, 400, 3])

        print('SHAPES:', self.all_c2w.shape, self.all_images.shape, self.all_fg_masks.shape, self.directions.shape)
        self.all_depths = None # depth is not provided in nerfies dataset

        self.h, self.w = self.all_images.shape[1:-1]


class iPhoneDatasetFromAllFrames(NerfiesDatasetFromAllFrames):

    __parser_cls__: Parser = iPhoneParser

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.training: # depth is not provided for the validation views
            depths = np.array([self.parser.load_depth(_t, _c) for _t, _c in zip(self._time_ids, self._camera_ids)])
            self.all_depths = torch.from_numpy(depths).to(self.rank)


class iPhoneDatasetFromAllFramesDataset(torch.utils.data.Dataset, iPhoneDatasetFromAllFrames):
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class iPhoneDatasetFromAllFramesIterableDataset(torch.utils.data.IterableDataset, iPhoneDatasetFromAllFrames):
    def __iter__(self):
        while True:
            yield {}

import pytorch_lightning as pl
import datasets

@datasets.register('iPhoneDatasetFromAllFrames_dataset')
class iPhoneDatasetFromAllFramesDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        kwargs = dict(dataset=self.config.dataset, sequence=self.config.scene, data_root=self.config.data_root)
        kwargs.update(self.config.get('dataset_kwargs', {}))
        if stage in [None, 'fit']:
            kwargs['split'] = self.config.train_split
            self.train_dataset = iPhoneDatasetFromAllFramesIterableDataset(**kwargs)
        if stage in [None, 'fit', 'validate']:
            kwargs['split'] = self.config.val_split
            self.val_dataset = iPhoneDatasetFromAllFramesDataset(**kwargs)
        if stage in [None, 'test']:
            kwargs['split'] = self.config.test_split
            self.test_dataset = iPhoneDatasetFromAllFramesDataset(**kwargs)
        # if stage in [None, 'predict']:
        #     self.predict_dataset = iPhoneDatasetFromAllFramesPredictDataset(**kwargs, split=self.config.test_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return torch.utils.data.DataLoader(
            dataset, 
            num_workers=0,#os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )

    @staticmethod
    def get_metadata(config):
        __parser_cls__ = iPhoneParser if config.dataset == 'iphone' else NerfiesParser
        parser = __parser_cls__(
            dataset=config.dataset,
            sequence=config.scene,
            data_root=config.data_root,
            # split=config.train_split
        )

        bkgd_points = parser.load_bkgd_points()
        pts_min = bkgd_points.min(0)
        pts_max = bkgd_points.max(0)
        
        return {
            'near': parser.near, 'far': parser.far,
            'time_max': int(parser.uniq_time_ids.max()),
            'train_img_hw': (480, 360),  # TODO: the resolution is temporary hardcoded
            # 'bkgd_points': bkgd_points,
            'scene_aabb': pts_min.tolist() + pts_max.tolist(),
        }

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       



if __name__ == "__main__":
    ds_train = iPhoneDatasetFromAllFrames(dataset='iphone', sequence='paper-windmill', split='train', data_root='/vlg-nfs/markomih/projects_dynamic/dystream/datasets')
    # ds_val = iPhoneDatasetFromAllFrames(dataset='iphone', sequence='paper-windmill', split='val', data_root='/vlg-nfs/markomih/projects_dynamic/dystream/datasets')

    # verify bg points 
    # unproject all depth sequences
    _bg_pts = ds_train.bkgd_points
    import trimesh
    from utils.ray_utils import get_rays

    trimesh.PointCloud(_bg_pts.detach().cpu().numpy()).export('paper-windmill_bg_pts.ply')

    # gen depth
    all_pts = []
    for cam_id in [50, 100, 150, 200]:
        rays_o, rays_d = get_rays(ds_train.directions, ds_train.all_c2w[cam_id], keepdim=False) # H*W,3

        depth = ds_train.all_depths[cam_id].reshape(-1)  # (HW,)
        dmask = depth > 0
        pts3d = (rays_o + rays_d*depth[:, None])[dmask]
        all_pts.append(pts3d)
    all_pts = torch.cat(all_pts, dim=0)
    trimesh.PointCloud(all_pts.detach().cpu().numpy()).export('paper-windmill_pts.ply')
