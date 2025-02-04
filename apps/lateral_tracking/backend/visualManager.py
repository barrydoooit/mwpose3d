import apps.lateral_tracking.constants as const

from .game_adapter import GameAdapter

class VisualManager:
    def __init__(self):
        self.visual = GameAdapter()

    def update(self, trackbuffer, frame_number=0, detObj=None):
        self.visual.update(trackbuffer, detObj=detObj, frame_number=frame_number)