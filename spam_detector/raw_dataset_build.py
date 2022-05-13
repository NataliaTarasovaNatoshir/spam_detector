import os
import shutil
import tarfile
import random

def flatten_folders(folder, levels_num):
    subdir_names = os.listdir(folder)
    if levels_num == 2:
        for subdir_name in subdir_names:
            emails = os.listdir(os.path.join(folder, subdir_name))
            for email in emails:
                shutil.move(os.path.join(folder, subdir_name, email),
                            os.path.join(folder, subdir_name + '_' + email))
    elif levels_num == 3:
        for subdir_name in subdir_names:
            subsubdir_names = os.listdir(os.path.join(folder, subdir_name))
            for subsubdir_name in subsubdir_names:
                emails = os.listdir(os.path.join(folder, subdir_name, subsubdir_name))
                for email in emails:
                    shutil.move(os.path.join(folder, subdir_name, subsubdir_name, email),
                                os.path.join(folder, subdir_name + '_' + subsubdir_name + '_' + email))
            for subsubdir_name in subsubdir_names:
                shutil.rmtree(os.path.join(folder, subdir_name, subsubdir_name), ignore_errors=True)
    else:
        raise Exception("{} is not supported levels number for flattening. Available levels: 2, 3".format(levels_num))
    for subdir_name in subdir_names:
        shutil.rmtree(os.path.join(folder, subdir_name), ignore_errors=True)


def build_raw_dataset(build_config):
    # define configuration
    res_dataset_folder_name = build_config['res_dataset_folder_name']
    raw_files_path = build_config['raw_files_folder']
    ham_files = build_config['ham_files']
    spam_files = build_config['spam_files']
    test_share = build_config['test_share']

    # prepare folder structure for dataset storage
    shutil.rmtree(res_dataset_folder_name, ignore_errors=True)
    os.makedirs(res_dataset_folder_name, exist_ok=False)
    for folder_name in ('train', 'test'):
        os.makedirs(os.path.join(res_dataset_folder_name, folder_name), exist_ok=False)
        os.makedirs(os.path.join(res_dataset_folder_name, folder_name, 'ham'), exist_ok=False)
        os.makedirs(os.path.join(res_dataset_folder_name, folder_name, 'spam'), exist_ok=False)

    # process each tar file in raw data
    raw_files = os.listdir(raw_files_path)
    for file_name in raw_files:
        print("processing {}".format(file_name))

        # un-tar file to a directory with the same name
        print("unpacking files")
        file = tarfile.open(os.path.join(raw_files_path, file_name))
        file.extractall(res_dataset_folder_name)
        file.close()
        src_folder_name = os.path.join(res_dataset_folder_name, file_name.split('.')[0])
        print('unpacked to {}'.format(src_folder_name))

        # flatten inner folder structure - move all files to the source directory
        print('unpack files to one directory')
        if file_name == "BG.tar.gz":
            flatten_folders(folder=src_folder_name, levels_num=3)
        else:
            flatten_folders(folder=src_folder_name, levels_num=2)

        # separate all documents from the source folder into train and test
        documents = os.listdir(src_folder_name)
        test_documents_num = int(len(documents) * test_share)
        print("total number of documents = {}, test documents number = {}".format(
            len(documents), test_documents_num))

        if file_name in ham_files:
            subdir_name = 'ham'
        elif file_name in spam_files:
            subdir_name = 'spam'
        else:
            raise Exception("Unknown file type. {} not in ham or spam group".format(file_name))

        print('add documents to {}'.format(subdir_name))
        print('move random sample to test')
        for doc_name in random.sample(documents, test_documents_num):
            shutil.move(os.path.join(src_folder_name, doc_name),
                        os.path.join(res_dataset_folder_name, 'test', subdir_name, doc_name))
        print('move other files to train')
        for doc_name in os.listdir(src_folder_name):
            shutil.move(os.path.join(src_folder_name, doc_name),
                        os.path.join(res_dataset_folder_name, 'train', subdir_name, doc_name))
        print('remove unpacked folder')
        shutil.rmtree(src_folder_name, ignore_errors=True)
        print()

    # show resulting statistics
    ham_test_cnt = len(os.listdir(os.path.join(res_dataset_folder_name, 'test', 'ham')))
    spam_test_cnt = len(os.listdir(os.path.join(res_dataset_folder_name, 'test', 'spam')))
    ham_train_cnt = len(os.listdir(os.path.join(res_dataset_folder_name, 'train', 'ham')))
    spam_train_cnt = len(os.listdir(os.path.join(res_dataset_folder_name, 'train', 'spam')))
    print('Resulting statistics')
    print('Total number of files = {0}. Train: {1}, Test: {2} ({3:.0f}%)'.format(
        ham_test_cnt + spam_test_cnt + ham_train_cnt + spam_train_cnt, ham_train_cnt + spam_train_cnt,
        ham_test_cnt + spam_test_cnt,
        100 * (ham_test_cnt + spam_test_cnt) / (ham_test_cnt + spam_test_cnt + ham_train_cnt + spam_train_cnt)))
    print("Train. Ham: {0}, Spam: {1} ({2:.0f}%)".format(ham_train_cnt, spam_train_cnt,
                                                         100 * spam_train_cnt / (ham_train_cnt + spam_train_cnt)))
    print("Test. Ham: {0}, Spam: {1} ({2:.0f}%)".format(ham_test_cnt, spam_test_cnt,
                                                         100 * spam_test_cnt / (ham_test_cnt + spam_test_cnt)))


