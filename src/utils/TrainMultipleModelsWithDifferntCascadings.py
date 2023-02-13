import os
import time

from src import CASCADE_DIR, OUTPUT_DIR
from src.TrainModel import ModelTrainer

if __name__ == "__main__":
    for filename in os.listdir(CASCADE_DIR):
        print(filename)
        if filename.endswith('.xml'):
            cascade = filename.split(".")[0]
            start_time = time.time()
            trainer = ModelTrainer(cascade_classifier=f'{CASCADE_DIR}/{cascade}.xml', debugging=True)
            trainer.train(output_path=f'{OUTPUT_DIR}/{cascade}_model')
            duration = time.time() - start_time
            print("Done in", str(duration) + "s")
