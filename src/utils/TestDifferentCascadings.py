import os

from src import CASCADE_DIR, TRAINDATA_DIR, OUTPUT_DIR
from src.recognizers.ImageFileGenericFaceRecognizer import ImageFileGenericFaceRecognizer

filenames = os.listdir(CASCADE_DIR)

for filename in filenames:
    if filename.endswith('.xml'):
        cascade = filename.split(".")[0]
        cascade_classifier = f'{CASCADE_DIR}/{cascade}.xml'
        Recognizer = ImageFileGenericFaceRecognizer(cascade_classifier=cascade_classifier, debugging=False)
        score = Recognizer.run(path=TRAINDATA_DIR + "/dominik_fladung")
        print(f'{filename} - Score: {str(score)}')

# haarcascade_eye.xml - Score: 49
# haarcascade_eye_tree_eyeglasses.xml - Score: 37
# haarcascade_frontalface_alt.xml - Score: 11
# haarcascade_frontalface_alt2.xml - Score: 15
# haarcascade_frontalface_alt_tree.xml - Score: -41
# haarcascade_frontalface_default.xml - Score: 31
# haarcascade_fullbody.xml - Score: -51
# haarcascade_lefteye_2splits.xml - Score: 47
# haarcascade_lowerbody.xml - Score: -47
# haarcascade_profileface.xml - Score: -51
# haarcascade_righteye_2splits.xml - Score: 47
# haarcascade_smile.xml - Score: 53
# haarcascade_upperbody.xml - Score: -41
