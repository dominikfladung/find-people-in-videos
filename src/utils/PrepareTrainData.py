"""
This class prepares the training data by renaming the files in the traindata folder
"""
import os

from src import TRAINDATA_DIR


class PrepareTrainData:
    @staticmethod
    def rename_files(filepath, suffix=""):
        """
        It takes a filepath as an argument, and renames all the files in that directory to a number,
        starting from 1
        
        :param suffix: filename suffix
        :param filepath: The path to the folder containing the files to be renamed
        """
        files = os.listdir(filepath)

        for i, file in enumerate(files):
            extension = "." + file.split(".")[-1].lower()
            target_filepath = os.path.join(filepath, str(i + 1) + suffix + extension)
            source_filepath = os.path.join(filepath, file)
            os.rename(source_filepath, target_filepath)

    def prepare_train_data(self):
        """
        It takes the path of the directory containing the training data, and renames the files in each
        subdirectory to the name of the subdirectory.
        """
        dirs = os.listdir(TRAINDATA_DIR)

        for dir in dirs:
            self.rename_files(TRAINDATA_DIR + "/" + dir, "tmp") # ensure no duplicate numbering
            self.rename_files(TRAINDATA_DIR + "/" + dir)


if __name__ == "__main__":
    PrepareTrainData().prepare_train_data()
