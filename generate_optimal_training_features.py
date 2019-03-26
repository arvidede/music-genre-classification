import sys
from numpy import load, std, mean, vstack, array, reshape, save
from sklearn import preprocessing
from operator import itemgetter
from constants import NUMBER_OF_FEATURES, NUMBER_OF_TRAINING_FEATURES, SIMILARITY_WEIGHT, SCORE_EXP, ALL_TRAINING_FEATURES_DATA_DIR, OPTIMAL_TRAINING_FEATURES_DIR, OPTIMAL_TRAINING_FEATURES_DATA_DIR
from scale_data import scale_data_single
from k_nearest_neighbor import find_optimal_k

def get_features_sorted_by_same_genre_similarity(all_training_features):
    genre_stds = []
    for feature_index in range(NUMBER_OF_FEATURES):
        genre_stds.append(({
            'min': [],
            'mean': [],
            'max': []
        }))

    for genre, features in all_training_features.items():
        for feature_index, feature in enumerate(features):
            for prop, prop_values in feature.items():
                genre_stds[feature_index][prop].append(std(prop_values))

    average_genre_stds = {
        'min': [0] * NUMBER_OF_FEATURES,
        'mean': [0] * NUMBER_OF_FEATURES,
        'max': [0] * NUMBER_OF_FEATURES
    }

    for feature_index, feature in enumerate(genre_stds):
        for prop, prop_stds in feature.items():
            average_genre_stds[prop][feature_index] = mean(prop_stds)

    scaled_average_genre_stds = {}
    for prop, stds in average_genre_stds.items():
        stds = array(stds).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)).fit(stds)
        scaled_average_genre_stds[prop] = scaler.transform(stds)

    feature_props = []
    for prop, scaled_average_genre_feature_stds in scaled_average_genre_stds.items():
        for feature_index, scaled_average_genre_feature_std in enumerate(scaled_average_genre_feature_stds):
            feature_id = feature_index + 1
            feature_name = str(feature_id) + '_' + prop
            feature_props.append({
                'feature_id': feature_id,
                'feature_prop': prop,
                'feature_name': feature_name,
                'std': scaled_average_genre_feature_std[0]
            })

    features_sorted_by_same_genre_similarity = sorted(feature_props, key=itemgetter('std'))
    return features_sorted_by_same_genre_similarity


def get_features_sorted_by_genre_difference(all_training_features):
    combined_genre_feature_prop_averages = []
    for feature_index in range(NUMBER_OF_FEATURES):
        combined_genre_feature_prop_averages.append({
            'min': [],
            'mean': [],
            'max': []
        })

    for genre, features in all_training_features.items():
        for feature_index, feature in enumerate(features):
            for prop, prop_values in feature.items():
                combined_genre_feature_prop_averages[feature_index][prop].append(mean(prop_values))

    feature_props = []
    for feature_index, feature in enumerate(combined_genre_feature_prop_averages):
        for prop, prop_averages in feature.items():
            feature_id = feature_index + 1
            feature_name = str(feature_id) + '_' + prop
            feature_props.append({
                'feature_id': feature_id,
                'feature_prop': prop,
                'feature_name': feature_name,
                'std': std(prop_averages)
            })

    features_sorted_by_genre_difference = sorted(feature_props, key=itemgetter('std'), reverse=True)
    return features_sorted_by_genre_difference

def get_sorted_features(features_sorted_by_same_genre_similarity, features_sorted_by_genre_difference):
    scored_features = []
    for index, feature in enumerate(features_sorted_by_same_genre_similarity):
        same_genre_similarity_score = (index ** SCORE_EXP) * SIMILARITY_WEIGHT
        genre_difference_score = (list(map(itemgetter('feature_name'), features_sorted_by_genre_difference)).index(feature['feature_name']) ** SCORE_EXP) * (1 - SIMILARITY_WEIGHT)
        scored_features.append({
            'feature_id': feature['feature_id'],
            'feature_prop': feature['feature_prop'],
            'score': same_genre_similarity_score + genre_difference_score
        })

    sorted_features = sorted(scored_features, key=itemgetter('score'))
    return sorted_features


def find_optimal_training_features(all_training_features):
    features_sorted_by_same_genre_similarity = get_features_sorted_by_same_genre_similarity(all_training_features)
    features_sorted_by_genre_difference = get_features_sorted_by_genre_difference(all_training_features)
    sorted_features = get_sorted_features(features_sorted_by_same_genre_similarity, features_sorted_by_genre_difference)
    return sorted_features[0:NUMBER_OF_TRAINING_FEATURES]

def save_optimal_training_features(all_training_features, optimal_training_features):
    optimal_training_features_data = {
        'X_train': [],
        'y_train': [],
        'k': 0
    }

    for genre, features in all_training_features.items():
        genre_X_train = []
        genre_y_train = []
        for song_index in range(len(features[0]['min'])):
            genre_X_train.append([])
            genre_y_train.append(genre)

        for feature_index, feature in enumerate(features):
            feature_id = feature_index + 1
            for prop, prop_values in feature.items():
                if any(f['feature_id'] == feature_id and f['feature_prop'] == prop for f in optimal_training_features):
                    for song_index, prop_value in enumerate(prop_values):
                        genre_X_train[song_index].append(prop_value)

        optimal_training_features_data['X_train'].extend(genre_X_train)
        optimal_training_features_data['y_train'].extend(genre_y_train)

    X_train_scale = scale_data_single(optimal_training_features_data['X_train'])
    optimal_training_features_data['k'] = find_optimal_k(X_train_scale, optimal_training_features_data['y_train'])

    save(OPTIMAL_TRAINING_FEATURES_DIR, optimal_training_features)
    save(OPTIMAL_TRAINING_FEATURES_DATA_DIR, optimal_training_features_data)


def main(argv):
    all_training_features = load(ALL_TRAINING_FEATURES_DATA_DIR).item()
    optimal_training_features = find_optimal_training_features(all_training_features)
    save_optimal_training_features(all_training_features, optimal_training_features)


if __name__ == "__main__":
    main(sys.argv)
