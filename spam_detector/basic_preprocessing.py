import os
import pandas as pd


def preprocess_message_text(message_text):
    res = {'subject': None, 'text': None}
    lines = message_text.split('\n')
    if lines[0][:len('Subject:')] == 'Subject:':
        res['subject'] = lines[0][len('Subject: '):]
        res['text'] = ' '.join(lines[1:])
    else:
        res['text'] = ' '.join(lines)
    return res


def preprocess_mails_in_folder(folder_path):
    print('Processing files in folder {}'.format(folder_path))
    df = []
    all_files = os.listdir(folder_path)
    print('Number of files in folder: {}'.format(len(all_files)))
    processed_num = 0
    for file_name in all_files:
        if processed_num%100 == 0: print("{} files out of {}".format(
            processed_num, len(all_files)))
        with open(os.path.join(folder_path, file_name), 'r') as f:
            text = f.read()
        preprocessed_text = preprocess_message_text(text)
        preprocessed_text['message_id'] = file_name
        df.append(preprocessed_text)
        processed_num += 1
    print('Preparing dataframe')
    df = pd.DataFrame(df)
    print('Processing completed')
    return df

