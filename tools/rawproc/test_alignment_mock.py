
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("e:/Projects/mwpose3d"))
sys.path.append(os.path.abspath("e:/Projects/mwCore"))

# Mock mwcore dependencies if they can't be imported, but since we are in the environment, we might be able to import them.
# However, to avoid dependency on physical files, we will mock RawBinReader.

from tools.rawproc.episode import Episode

class TestAlignment(unittest.TestCase):
    
    @patch('tools.rawproc.episode.RawBinReader')
    def test_load_pcd_bin(self, MockReader):
        # Setup mock reader
        mock_reader_instance = MockReader.return_value
        mock_reader_instance.current_frame_idx = 1
        
        # Mock read() to return 2 frames then None
        # Frame 1: ts=100.0, 5 points
        pcd1 = np.zeros((6, 5)) 
        pcd1[0, :] = 1.0 # x
        pcd1[4, :] = 10.0 # energy/snr
        
        # Frame 2: ts=100.1, 3 points
        pcd2 = np.zeros((6, 3))
        pcd2[0, :] = 2.0
        
        mock_reader_instance.read.side_effect = [
            (100.0, pcd1),
            (100.1, pcd2),
            None # End of file
        ]
        
        # Mock close
        mock_reader_instance.close.return_value = None
        
        # Create Episode
        episode = Episode("test_episode")
        
        # Create dummy bin file path (doesn't need to exist because we mock RawBinReader, 
        # but Episode checks existence)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            bin_path = Path(tf.name)
        
        try:
            episode.load_pcd_bin(bin_path)
            
            # Verify pcd_df
            self.assertIsNotNone(episode.pcd_df)
            self.assertEqual(len(episode.pcd_df), 5 + 3)
            
            # Check content
            df = episode.pcd_df
            self.assertTrue('ts' in df.columns)
            self.assertTrue('x' in df.columns)
            
            # Timestamp scaling (seconds -> ms)
            # data was 100.0, 100.1 -> 100000, 100100
            self.assertEqual(df.iloc[0]['ts'], 100000) 
            self.assertEqual(df.iloc[5]['ts'], 100100)
            
            # Check if seq updated (based on side effects logic in load_pcd_bin)
            # logic in load_pcd_bin: 'seq': reader.current_frame_idx - 1
            # We didn't mock current_frame_idx incrementing.
            # Real RawBinReader increments it.
            # We should mock property or update it in side_effect? 
            # Too complex for quick mock.
            
        finally:
            if bin_path.exists():
                os.remove(bin_path)

if __name__ == '__main__':
    unittest.main()
