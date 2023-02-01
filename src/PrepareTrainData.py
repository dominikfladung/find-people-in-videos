import os


class PrepareTrainData:
    @staticmethod
    def rename_files(filepath):
        files = os.listdir(filepath)

        for i, file in enumerate(files):
            os.rename(os.path.join(filepath, file),
                      os.path.join(filepath, str(i + 1) + '.jpeg'))

    def prepare_train_data(self):
        path = "traindata"
        dirs = os.listdir(path)
        for dir in dirs:
            self.rename_files(path + "/" + dir)


if __name__ == "__main__":
    PrepareTrainData().prepare_train_data()
