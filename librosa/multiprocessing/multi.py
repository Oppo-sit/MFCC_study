import time
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.cm as cm
import pandas as pd
import joblib




def find_emotion(csvfile, str):
    return csvfile[csvfile['wav_id'].str.contains(str)].iloc[0][1]


def mfcc_multi(y):

    # 경로 확장자 분리 참조 : https://jvvp.tistory.com/980
    y = ''.join(y) # y가 튜플타입이기 때문에 문자열로 변환해주어야함.
    print(y)
    audio_path = path + '/' + y
    file_code, ext = os.path.splitext(y)  # 확장자 분리
    dirname, basename = os.path.split(file_code)  # 경로 분리
    y, sr = librosa.load(audio_path, 16000)

    frame_length = 0.025
    frame_stride = 0.01
    sr = 16000

    n_fft_25 = int(round(sr * frame_length))
    hop_length_10 = int(round(sr * frame_stride))

    n_fft = n_fft_25 * 2
    win_length = n_fft_25
    hop_length = hop_length_10
    n_mels = 128  # default
    n_mfcc = 20  # default

    # mfcc에는 멜 스펙트로그램의 log값이 필요하다.
    ## 원래는 logamplitude 함수로 변환하였으나, power를 db로 변환하는 과정 자체가 로그함수를 씌우는 것이기 때문에 power_to_db로 함수가 대체됨

    # audio file -> MFCC data
    D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
    MS = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                                        win_length=win_length)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(D), sr=sr, n_mfcc=n_mfcc)

    # numpy array csv로 추출 참조 : https://appia.tistory.com/204
    if find_emotion(pd_csv_id, basename) == 'anger':
        ext_name = path + '/res_anger/' + basename + '.csv'
        np.savetxt(ext_name, mfccs, delimiter=",")
    elif find_emotion(pd_csv_id, basename) == 'sad':
        ext_name = path + '/res_sad/' + basename + '.csv'
        np.savetxt(ext_name, mfccs, delimiter=",")
    elif find_emotion(pd_csv_id, basename) == 'disgust':
        ext_name = path + '/res_disg/' + basename + '.csv'
        np.savetxt(ext_name, mfccs, delimiter=",")
    elif find_emotion(pd_csv_id, basename) == 'fear':
        ext_name = path + '/res_fear/' + basename + '.csv'
        np.savetxt(ext_name, mfccs, delimiter=",")
    else:
        ext_name = path + '/result/' + basename + '.csv'
        np.savetxt(ext_name, mfccs, delimiter=",")


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


if __name__ == "__main__":
    path = 'C:/Users/RestJSM/librosa/data/emotion_dataset/test'
    audio_list = os.listdir(path)

    createFolder(path + '/res_anger')
    createFolder(path + '/res_sad')
    createFolder(path + '/res_neutral')
    createFolder(path + '/res_disg')
    createFolder(path + '/res_fear')

    pd_path = 'C:/Users/RestJSM/librosa/data/emotion_dataset'
    pd_csv = pd.read_csv(pd_path + '/5th_year.csv', encoding='cp949')
    pd_csv_id = pd_csv.loc[:, ['wav_id', '상황']]
    # print(pd_csv_id)


    n_jobs = 4
    verbose = 1

    jobs = [joblib.delayed(mfcc_multi)(audio) for audio in zip(audio_list)]
    out = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    print(out)
