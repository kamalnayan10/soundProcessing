import os
import librosa
import math
import json

DATASET_PATH = "D:\PROGRAMMING\PYTHON\genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050

DURATION = 30 #measured in seconds

SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc = 13 , n_fft = 2048 , hop_length = 512 , num_segments = 5):
    # Build Dictionary to store data
    data = {
        "mapping":[],
        "mfcc":[],
        "labels":[]
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)

    # Loop thorugh all the genres

    for i , (dirpath , dirnames ,filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure that we're not at the root level
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/") #genres/blues -> ['genres , 'blues']
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            print(f"\n Processing {semantic_label}")

            # Process files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath , f)
                signal , sr = librosa.load(file_path , sr = SAMPLE_RATE)

                #Process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment*s
                    finish_sample = start_sample + num_samples_per_segment

                # Store mfcc for segment if it has expected length
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample] , sr = sr , n_mfcc = n_mfcc , n_fft = n_fft , hop_length = hop_length)

                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print(f"{file_path} , {s}")

    with open(JSON_PATH , "w") as fp:
        json.dump(data , fp)

if __name__ == "__main__":
    
    save_mfcc(DATASET_PATH , JSON_PATH , num_segments = 10)
    