from pathlib import Path
import pickle


def main():
    train_val_test_rario = (6, 1, 1)
    data_root = Path('./data/neat')
    assert data_root.exists(), f"Data root directory {data_root} does not exist."
    info_all_f = 'info_all.pkl'
    train_info_f = 'info_train.pkl'
    test_info_f = 'info_test.pkl'
    val_info_f = 'info_val.pkl'

    with open(data_root / info_all_f, 'rb') as f:
        info_all = pickle.load(f)
    
    files = list(info_all.keys())
    train_info = {}
    test_info = {}
    val_info = {}

    for file in files:
        info = info_all[file]
        frame_count = info['frame_count']
        train_frames = frame_count // sum(train_val_test_rario) * train_val_test_rario[0]
        test_frames = frame_count // sum(train_val_test_rario) * train_val_test_rario[2]
        val_frames = frame_count - train_frames - test_frames

        train_info[file] = dict(info, **{
            'snippets': [(0, train_frames)],
            'frame_count': train_frames})
        test_info[file] = dict(info, **{
            'snippets': [(train_frames, train_frames + test_frames)],
            'frame_count': test_frames})
        val_info[file] = dict(info, **{
            'snippets': [(train_frames + test_frames, frame_count)],
            'frame_count': val_frames})
    
    with open(data_root / train_info_f, 'wb') as f:
        pickle.dump(train_info, f)
    with open(data_root / test_info_f, 'wb') as f:  
        pickle.dump(test_info, f)
    with open(data_root / val_info_f, 'wb') as f:
        pickle.dump(val_info, f)

if __name__ == '__main__':
    main()
    # with open('./data/neat/info_train.pkl', 'rb') as f:
    #     info = pickle.load(f)
    # print(info)