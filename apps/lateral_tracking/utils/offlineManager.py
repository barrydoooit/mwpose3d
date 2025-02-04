import csv
import os
import apps.lateral_tracking.constants as const



class OfflineManager:
    """
    A class for managing the reading of frames from an offline experiment file.

    Attributes
    ----------
    experiment_path : str
        The path to the directory containing the experiment files.
    frame_count : int
        The count of frames read so far.
    pointer : List[int]
        A list containing two integers: the index of the last read frame and the index of the current file being read.
    pointclouds : Dict[int, Dict[str, List[float]]]
        A dictionary containing point cloud data for each frame.
    last_frame : int or None
        The index of the last frame read from the experiment file, or None if the experiment has finished.

    Methods
    -------
    read_next_frames()
        Read the next batch of frames from the experiment file.
    get_data()
        Get the data for the current frame.
    is_finished() -> bool
        Check if the offline experiment has finished.
    """

    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.frame_count = 0
        self.pointer = [0, 1]
        self.read_next_frames()

    def read_next_frames(self):
        """
        Read the next batch of frames from the given experiment file starting from the specified frame number.
        """
        self.pointclouds = {}
        self.last_frame = None

        while len(self.pointclouds) < const.FB_READ_BUFFER_SIZE:
            file_path = os.path.join(self.experiment_path, f"{self.pointer[1]}.csv")
            try:
                with open(file_path, "r") as file:
                    csv_reader = csv.reader(file)
                    for index, row in enumerate(csv_reader):
                        # Pass previously parsed frames
                        if index < self.pointer[0]:
                            continue

                        framenum = int(row[0])
                        coords = [
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                            int(row[6]),
                        ]

                        # Read only the frames in the specified range
                        if framenum in self.pointclouds:
                            # Append coordinates to the existing lists
                            for key, value in zip(
                                ["x", "y", "z", "doppler", "peakVal", "posix"], coords
                            ):
                                self.pointclouds[framenum][key].append(value)
                        else:
                            # If not, create a new dictionary for the framenum
                            self.pointclouds[framenum] = {
                                "x": [coords[0]],
                                "y": [coords[1]],
                                "z": [coords[2]],
                                "doppler": [coords[3]],
                                "peakVal": [coords[4]],
                                "posix": [coords[5]],
                            }

                        self.last_frame = framenum

                        if len(self.pointclouds) >= const.FB_READ_BUFFER_SIZE:
                            # Break the loop once const.FB_READ_BUFFER_SIZE frames are read
                            self.pointer[0] = index + 1
                            break
                    else:
                        self.pointer[0] = 0
                        self.pointer[1] += 1

            except FileNotFoundError:
                break

    def get_data(self):
        """
        Get the data for the current frame.

        Returns
        -------
        exists : bool
            True if data for the current frame exists, False otherwise.
        frame_count : int
            The count of frames read so far.
        data : dict or None
            The point cloud data for the current frame, or None if data for the frame is not available.
        """

        self.frame_count += 1
        # If the read buffer is parsed, read more frames from the experiment file
        if self.frame_count > self.last_frame:
            self.read_next_frames()

        if self.frame_count in self.pointclouds:
            return True, self.frame_count, self.pointclouds[self.frame_count]
        else:
            return False, self.frame_count, None

    def is_finished(self):
        """
        Check if the offline experiment has finished.

        Returns
        -------
        bool
            True if the experiment has finished (last_frame is None), False otherwise.
        """
        return self.last_frame is None