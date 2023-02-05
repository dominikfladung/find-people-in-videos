import os
from src.detectors.ImageFileGenericFaceDetector import ImageFileGenericFaceDetector

filenames = os.listdir('../../cascades/data')

for filename in filenames:
    if filename.endswith('.xml'):
        cascade = filename.split(".")[0]
        cascade_classifier = f'../../cascades/data/{cascade}.xml'
        detector = ImageFileGenericFaceDetector(cascade_classifier=cascade_classifier, debugging=False)
        score = detector.run(path="../../traindata/dominik_fladung", model_path=f'../../output/{cascade}_model.xml')
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
