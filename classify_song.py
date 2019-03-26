import os
import sys
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from numpy import mean, load
import k_nearest_neighbor
from operator import itemgetter
from constants import OPTIMAL_TRAINING_FEATURES_DIR, OPTIMAL_TRAINING_FEATURES_DATA_DIR, DATASET_DIR
import pafy
from pydub import AudioSegment
from scale_data import scale_data_multiple

def predict_genre(song_path):
    optimal_training_features = load(OPTIMAL_TRAINING_FEATURES_DIR)
    training_data = load(OPTIMAL_TRAINING_FEATURES_DATA_DIR).item()

    X_train = training_data['X_train']
    y_train = training_data['y_train']
    x_test = []

    Fs, x = audioBasicIO.readAudioFile(song_path)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

    features_to_extract = sorted(optimal_training_features, key=itemgetter('feature_id'))
    for feature in features_to_extract:
        feature_index = feature['feature_id'] - 1
        feature_prop = feature['feature_prop']
        feature_data = None
        if feature_prop == 'min':
            feature_data = min(F[feature_index, :])
        elif feature_prop == 'mean':
            feature_data = mean(F[feature_index, :])
        elif feature_prop == 'max':
            feature_data = max(F[feature_index, :])
        x_test.append(feature_data)

    X_train_scale, x_test_scale = scale_data_multiple(X_train, x_test)

    prediction = k_nearest_neighbor.k_nearest_neighbor(X_train_scale, y_train, x_test_scale, training_data['k'])
    return prediction

def get_songpath():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'url':
            video = pafy.new(input('Enter url: '))
            audio = video.getbestaudio(preftype='m4a')
            audio.download(filepath='./downloads/')
            m4a_path = os.path.join('./downloads/', video.title + '.m4a')
            m4a_version = AudioSegment.from_file(m4a_path, format='m4a')[30000:60000].set_channels(1).set_frame_rate(22050)
            mp3_version = m4a_version.export('./downloads/' + video.title + '.mp3', format='mp3')
            return os.path.join('./downloads/', video.title + '.mp3')
        elif arg == 'system':
            return sys.argv[2]
    else:
        return './datasets/GTZAN/genres/pop/pop.00001.au'
        print('You must specify url or system path')


def main(argv):
    path = get_songpath()
    genre = predict_genre(path)
    print('Predicted genre is: ' + genre)

if __name__ == "__main__":
    main(sys.argv)
