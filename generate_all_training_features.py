import os
import sys
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from numpy import mean, save
from constants import NUMBER_OF_FEATURES, DATASET_DIR, DATASET_FILETYPE, ALL_TRAINING_FEATURES_DATA_DIR

def genere_all_training_features():

        genre = None
        all_training_features_data = {}

        for subdir, dirs, files in os.walk(DATASET_DIR):
            genre = subdir.split('/')[-1]
            genre_features = None

            if genre:
                genre_features = []
                for feature_index in range(NUMBER_OF_FEATURES):
                    genre_features.append({
                        'min': [],
                        'mean': [],
                        'max': []
                    })

            for file in files:
                if file.split('.')[-1] == DATASET_FILETYPE:
                    path = os.path.join(subdir, file)
                    Fs, x = audioBasicIO.readAudioFile(path)
                    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)

                    for feature_index in range(NUMBER_OF_FEATURES):
                        genre_features[feature_index]['min'].append(min(F[feature_index, :]))
                        genre_features[feature_index]['mean'].append(mean(F[feature_index, :]))
                        genre_features[feature_index]['max'].append(max(F[feature_index, :]))

            if genre:
                all_training_features_data[genre] = genre_features

        save(ALL_TRAINING_FEATURES_DATA_DIR, all_training_features_data)


def main(argv):
    genere_all_training_features()

if __name__ == '__main__':
    main(sys.argv)
