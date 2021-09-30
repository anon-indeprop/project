import os
import json

basepath = "dataset/new/"
train_path = basepath + "train"
dev_path = basepath + "dev"
test_path = basepath + "test"

def process_split(split_folder):
    print(split_folder + " STARTED")
    files = os.listdir(split_folder)

    more_data = []

    for f in files:
        if not f.endswith('.json'):
            continue
        this_data = json.load(open(split_folder + '/' + f))
        for sentence in this_data['new sentences']:
            if sentence['new_text'] != None and sentence['new_text'].strip() != "":
                tokens = [x.strip() for x in sentence['new_text'].split()]
                more_data.append(tokens)

    json.dump(more_data, open(split_folder + ".json", 'w+'), indent=2)
    print(split_folder + " DONE")
 
process_split(train_path)
process_split(dev_path)
process_split(test_path)



