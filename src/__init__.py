import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.sep.join(SRC_DIR.split(os.path.sep)[:-1])
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
CASCADE_DIR = os.path.join(ROOT_DIR, 'cascades')
TRAINDATA_DIR = os.path.join(ROOT_DIR, 'traindata')
VIDEOS_DIR = os.path.join(ROOT_DIR, 'videos')

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if not os.path.isdir(VIDEOS_DIR):
    os.mkdir(VIDEOS_DIR)

DEFAULT_IMAGES_PATH = os.path.join(TRAINDATA_DIR, 'emilia_clarke')
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, 'model')
