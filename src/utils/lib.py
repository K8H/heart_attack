import os


def if_not_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
