from typing import Dict, List, Optional, Union
import numpy as np


class SimplePoint3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        
class SimplePoint5D(SimplePoint3D):
    def __init__(self, x: float, y: float, z: float, vel: float, snr: float):
        super().__init__(x, y, z)
        self.vel = vel
        self.snr = snr
    
    @staticmethod
    def from_numpy(data: np.ndarray):
        return SimplePoint5D(data[0], data[1], data[2], data[3], data[4])

    def serialize(self, compact=False)-> Union[List[float], dict]:
        if compact:
            return [self.x, self.y, self.z, self.vel, self.snr]
        else:
            return {'x': self.x, 'y': self.y, 'z': self.z, 'vel': self.vel, 'snr': self.snr}

class SimplePointCloud5D:
    def __init__(self, points: List[SimplePoint5D],):
        self.points = points

    @staticmethod
    def from_numpy(data: np.ndarray):
        return SimplePointCloud5D([SimplePoint5D.from_numpy(point) for point in data])
    
    @staticmethod
    def from_dict(data: Dict[str, np.ndarray]):
        if 'doppler' in data:
            data['vel'] = data.pop('doppler')
        if 'peakVal' in data:
            data['snr'] = data.pop('peakVal')
        data = np.stack([data[key] for key in ['x', 'y', 'z', 'vel', 'snr']], axis=-1)
        return SimplePointCloud5D.from_numpy(data)
        
    def serialize(self, compact=False)-> Union[List[List[float]], List[dict]]:
        return [point.serialize(compact) for point in self.points]