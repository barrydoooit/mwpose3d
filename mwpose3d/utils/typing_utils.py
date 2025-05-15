# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in MMDetection3D."""
from typing import List, Optional, Union

from mmengine.config import ConfigDict

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]