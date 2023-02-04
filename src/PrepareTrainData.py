import os


# This class prepares the training data by renaming the files in the traindata folder
class PrepareTrainData:
    @staticmethod
    def rename_files(filepath):
        """
        It takes a filepath as an argument, and renames all the files in that directory to a number,
        starting from 1
        
        :param filepath: The path to the folder containing the files to be renamed
        """
        files = os.listdir(filepath)

        for i, file in enumerate(files):
            os.rename(os.path.join(filepath, file),
                      os.path.join(filepath, str(i + 1) + '.jpeg'))

    def prepare_train_data(self):
        """
        It takes the path of the directory containing the training data, and renames the files in each
        subdirectory to the name of the subdirectory.
        """
        path = "traindata"
        dirs = os.listdir(path)
        for dir in dirs:
            self.rename_files(path + "/" + dir)


if __name__ == "__main__":
    PrepareTrainData().prepare_train_data()
