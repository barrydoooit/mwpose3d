import collections
from numpy import average, dot
from ..tracking import Status, TrackBuffer

from .breakout import Breakout
from .adapter import Adapter

import apps.lateral_tracking.constants as const
from .breakout import constants as game_const

class GameAdapter(Adapter):
    def __init__(self) -> None:
        self.current_player_pos = 0
        self.breakout = Breakout()
        self.breakout.start()

        l = const.MOVING_AVERAGE_WINDOW if const.MOVING_AVERAGE else 1
        self.x = collections.deque(maxlen=l)

    def update(self, trackbuffer: TrackBuffer, **kwargs):
        if trackbuffer.effective_tracks:
            track = trackbuffer.effective_tracks[0]
            if track:
                self.x.append(track.state.x[0])

                new_player_pos = average(self.x)

                displacement_meters = new_player_pos - self.current_player_pos

                # In case the player is moving, but we are inbetween frames, we need to interpolate.
                if const.INTERPOLATION and displacement_meters == 0 and track.track_status == Status.DYNAMIC:
                    # Interpolate based on predicted position.
                    x = track.state.x
                    F = const.MOTION_MODEL.KF_F(trackbuffer.dt / const.REFRESH_RATE_COEF)
                    new_player_pos = dot(F, x)
                    new_player_pos = new_player_pos[0]

                    displacement_meters = new_player_pos - self.current_player_pos

                displacement_ratio = displacement_meters / const.PLAYGROUND_WIDTH

                displacement_pixels = displacement_ratio * game_const.SCREEN_WIDTH

                self.breakout.move(displacement_pixels)
                
                self.current_player_pos = new_player_pos
            else:
                self.breakout.move(0)
        else: 
            self.breakout.move(0)