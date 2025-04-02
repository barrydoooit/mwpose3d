from torch import Tensor


def ball_query(new_xyz: Tensor, xyz: Tensor, radius: float, nsample: int) -> Tensor: ...
