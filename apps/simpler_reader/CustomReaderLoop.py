from apps.common.loops.onlineReader import OnlineReaderLoop


class CustomReaderLoop(OnlineReaderLoop):
    def _process_data(self, data):
        data_ok, frame_number, det_obj = data
        print(frame_number)