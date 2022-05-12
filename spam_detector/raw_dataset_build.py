import os


def build_raw_dataset(raw_files_path, target_path):
    print("search directory {}".format(raw_files_path))
    raw_files = os.listdir(raw_files_path)
    print("found raw files: {}".format(raw_files))

