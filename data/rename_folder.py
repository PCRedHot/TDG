import os

FILE_PATH = os.path.dirname(__file__)

TRAINING_PATH = os.path.join(FILE_PATH, 'training')
TESTING_PATH = os.path.join(FILE_PATH, 'testing')

for f in os.listdir(TRAINING_PATH):
    os.rename(
        os.path.join(TRAINING_PATH, f),
        os.path.join(TRAINING_PATH, f.replace('Edited ', '')),
    )
    
for f in os.listdir(TESTING_PATH):
    os.rename(
        os.path.join(TESTING_PATH, f),
        os.path.join(TESTING_PATH, f.replace('Edited ', '')),
    )