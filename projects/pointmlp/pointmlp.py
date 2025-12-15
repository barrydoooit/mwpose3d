import numpy as np
import torch
from torch import nn
from mmengine.device import get_device

from .pointmlp_impl import Model as PointMLP
from mwpose3d.models.base import BaseSkeletonEstimModel
from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.registry import MODELS


@MODELS.register_module()
class PointMLPPredictor(BaseSkeletonEstimModel):

    def __init__(
        self,
        point_cloud_size: int = 64,
        in_channels: int = 5,
        keypoints_involved: list[int] = list(range(0, 21)),
    ):
        super().__init__()
        self.point_cloud_size = point_cloud_size
        self.in_channels = in_channels
        self.keypoints_involved = keypoints_involved
        self.num_keypoints = len(keypoints_involved)
        self.criterion = nn.MSELoss()
        self.model = self.make_model()

    def make_model(self):
        # Same parameters as WixUp
        return PointMLP(
            points=self.point_cloud_size,
            class_num=self.num_keypoints * 3,
            normalize="anchor",
            k_neighbors=[12, 12, 12, 8], # 64 -> 32 -> 16 -> 8, cannot do 12 for the last layer
            bias=False
        )

    def loss(self, batch_inputs: dict, data_samples):
        result = self._forward(batch_inputs, data_samples)

        gt = torch.stack([data_sample.gt for data_sample in data_samples], dim=0)
        loss = self.criterion(result, gt)
        return loss

    def predict(self, batch_inputs: dict, data_samples):
        tensor = self._forward(batch_inputs, data_samples)
        for b, data_sample in enumerate(data_samples):
            data_sample.pred = tensor[b]
        return tensor

    def _forward(self, batch_inputs: dict, _):
        x = batch_inputs["final_pcd_tensor"]
        # print(f"{x.shape=}")
        return self.model(x)

    def pack_input(self, data_batch_dict: dict):
        # The code below is copied and adapted from the MARS predictor
        pcd_frame_list: list[tuple[np.ndarray]] = data_batch_dict[
            "pcd_frames"
        ]  # F x B x N x C
        batch_size = len(pcd_frame_list[0])
        frame_len = len(pcd_frame_list)
        assert frame_len == 1, "PointMLP is not a temporal model, we do not support multiple frames as input!"
        final_pcd_frame = np.zeros(
            (frame_len, batch_size, self.point_cloud_size, self.in_channels),
            dtype=np.float32,
        )

        for frame_seq, batched_frames in enumerate(pcd_frame_list):
            for batch_idx, pcd_frame in enumerate(batched_frames):
                final_pcd_frame[frame_seq, batch_idx] = pcd_frame[
                    : self.point_cloud_size
                ]

        final_pcd_tensor = torch.from_numpy(final_pcd_frame).float().to(get_device())
        final_pcd_tensor = final_pcd_tensor.permute(
            0, 1, 3, 2  # B x F x N x C --> B x F x C x N (Swap point and channels)
        )[0, :, :, :].contiguous()  # B x F x C x D --> B x C x N (Drop the frame dim of 1)

        skel_frame_list: list[tuple[np.ndarray]] = data_batch_dict["skel_frames"]

        last_skel_frame = (
            torch.from_numpy(np.stack(skel_frame_list[-1], axis=0))
            .float()
            .to(get_device())
        )
        data_sample_list = [
            SkeletonDataSample(
                gt=last_skel_frame_tensor[: (last_skel_frame_tensor.shape[0] // 3) * 3]
            )
            for last_skel_frame_tensor in last_skel_frame
        ]
        data_batch_dict["final_pcd_tensor"] = final_pcd_tensor
        return data_batch_dict, data_sample_list
