import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
OUTPUT_DIR = f'{ROOT_DIR}/output'
CASCADE_DIR = f'{ROOT_DIR}/cascades'
TRAINDATA_DIR = f'{ROOT_DIR}/traindata'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
