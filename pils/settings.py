import os

# Folders structures
HERE = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
ROOT = os.path.join(HERE, os.pardir)
PROBLEMS_FOLDER = os.path.join(HERE, "problems")
BIN_FOLDER = os.path.join(HERE, "bin")
TEMP_FOLDER = os.path.join(HERE, "temp")


def clean_lines(lines):
    return list(map(lambda f: f.replace("\n", ""), lines))
