NOTE: If you don't want to download the dataset and generate the features yourself, you can skip step 1 and 2 as well as A and B, and you will use the pre-generated files. You won't be able to change any parameters though, since the dataset is required to generate new features based on thoe parameters.


How to get started with Music Genre Classification

1. Download the GTZAN Genre Collection dataset from http://marsyasweb.appspot.com/download/data_sets/

2. Unarchive the compressed dataset and put it in the project folder using the following folder structure: <project_folder>/datasets/GTZAN/genres/...

3. Install all the required pip packages:
    - numpy
    - pafy
    - pydub
    - sklearn
    - matplotlib
    - scipy
    - hmmlearn
    - simplejson
    - eyed3
    - youtube-dl

4. Install additional required packages using brew (or something else):
    - libmagic
    - ffmpeg


How to use the classifier

A. To generate all possible training features, run the following:
    [python or python3] generate_all_training_features.py

B. To generate optimal training features based on the constants set in constants.py, run the following:
    [python or python3] generate_optimal_training_features.py

C. To run the classifier using a local file, run the following:
    [python or python3] classify_song.py system <file_path>

D. To run the classifier using a YouTube URL, run the following:
    [python or python3] url
and enter the URL when asked


### Authors:

Gustav Fridell
Arvid Edenheim
Saam Cedighi Chafjiri
