import os
import time

from src.TrainModel import ModelTrainer

for filename in os.listdir('../../cascades/data'):
    print(filename)
    if filename.endswith('.xml'):
        cascade = filename.split(".")[0]
        start_time = time.time()
        trainer = ModelTrainer(cascade_classifier=f'../../cascades/data/{cascade}.xml')
        trainer.train(dataset_path='../../traindata')
        trainer.save(path=f'../../output/{cascade}_model.xml')
        duration = time.time() - start_time
        print("Done in", str(duration) + "s")