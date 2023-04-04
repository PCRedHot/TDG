import os, shutil, random, math
from PIL import Image

FILE_PATH = os.path.dirname(__file__)

print(__file__)
print(FILE_PATH)

testing_proportion = 0.1

TRAINING_PATH = os.path.join(FILE_PATH, 'training')
TESTING_PATH = os.path.join(FILE_PATH, 'testing')

classes = os.listdir(TRAINING_PATH)

for class_name in classes:
    FROM_PATH = os.path.join(TRAINING_PATH, class_name)
    TO_PATH = os.path.join(TESTING_PATH, class_name)

    if not os.path.exists(TO_PATH):    
        os.makedirs(TO_PATH)

    # Move all images back to training path
    to_path_files = os.listdir(TO_PATH)
    for f in to_path_files:
        shutil.move(
            os.path.join(TO_PATH, f),
            os.path.join(FROM_PATH, f),
        )


    
    # # Randomise
    from_path_files = os.listdir(FROM_PATH)
    for f in from_path_files:
        try:
            img = Image.open(os.path.join(FROM_PATH, f))
            img.close()
        except:
            os.remove(os.path.join(FROM_PATH, f))


    to_testing_files = random.sample(from_path_files, math.floor(len(from_path_files) * testing_proportion))
    # print(to_testing_files)
    for f in to_testing_files:
        shutil.move(
            os.path.join(FROM_PATH, f),
            os.path.join(TO_PATH, f),
        )