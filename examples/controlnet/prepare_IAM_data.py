import os
from PIL import Image
import numpy as np
import pickle
# from datasets import Dataset

train_data_dir = '/media/ahmad/Windows7_OS/Downloads/IAM Handwritten'

sentences_data = dict()

with open(os.path.join(train_data_dir, 'ascii', 'sentences.txt'), 'r') as ascii_file:
    for i, line in enumerate(ascii_file):
        if i < 25:
            continue
        ascii_info = line.split()
        if ascii_info[2] != 'ok':
            continue
        sentence_id = ascii_info[0]

        tmp = ascii_info[0].split('-')
        form_id = '_'.join(tmp[:2])
        
        tmp = ascii_info[-1].split('|')
        if tmp[-1] in ('.', ',', '"', "'", ';', ':', '?', '!'):
            text = ' '.join(tmp[:-1]) + tmp[-1]
        else:
            text = ' '.join(tmp)

        sentences_data[sentence_id] = {'form_id': form_id, 'text': text}


cnt = 0
for root, dirs, files in os.walk(os.path.join(train_data_dir, 'sentences')):
    for file_name in files:
        sentence_id = file_name[:-4]
        if sentence_id not in sentences_data:
            continue

        cnt += 1
        if np.mod(cnt, 1000) == 0:
            print(cnt)
        file_path = os.path.join(root, file_name)
        try:
            with Image.open(file_path) as im:
                image = im.copy()
        except Exception as e:
            print(f"Failed to load image: {file_path}\nError: {str(e)}")
        sentences_data[sentence_id]['image'] = image

sentences_data = {k: v for k, v in sentences_data.items() if 'image' in v}

text_count = dict()
for sentence_id, sd in sentences_data.items():
    if sd['text'] in text_count:
        text_count[sd['text']] += 1
    else:
        text_count[sd['text']] = 1

forms_data = {}
for v in sentences_data.values():
    f_id = v['form_id']
    if f_id not in forms_data:
        forms_data[f_id] = []
    forms_data[f_id].append({'text': v['text'], 'image': v['image']})

data_dict = {'train': {"target_image": [], "conditioning_image": [], "prompt": []}}

cnt = 0
for f_id, s_list in forms_data.items():
    cnt += 1
    print(f'{cnt} / {len(forms_data)}')
    for i in range(len(s_list)):
        for j in range(len(s_list)):
            if s_list[i]['text'] == s_list[j]['text']: # i == j:
                continue
            data_dict['train']['target_image'].append(s_list[i]['image'])
            data_dict['train']['conditioning_image'].append(s_list[j]['image'])
            data_dict['train']['prompt'].append(f'target text: {s_list[i]["text"]} --- conditioning text: {s_list[j]["text"]} --- \
                                                target size: {s_list[i]["image"].size} --- conditioning size: {s_list[j]["image"].size}')

with open(os.path.join(train_data_dir, 'IAM-sentences-contnet-data.pkl'), 'wb') as fout:
    pickle.dump(data_dict, fout)

# print(np.unique(list(text_count.values())))
        
# for k, v in text_count.items():
#     if v > 1:
#         print(k + '      ', str(v))

# values = np.array(list(text_count.values())) 
# for t in range(np.max(values)):
#     print(f'{t+1}:\t{(values==t+1).sum()}')


# data_dict = {'train': {"pixel_values": [], "input_ids": [], "conditioning_pixel_values": []}}
# for text, cnt in text_count.items():
#     if cnt < 2:
#         continue
#     list_imgs = [v['image'] for v in sentences_data.values() if v['text'] == text]
#     for i in range(len(list_imgs)):
#         for j in range(i+1, range(len(list_imgs))):
#             data_list['train']['pixel_values'].

# print([v['image'].size for v in sentences_data.values()])
# print(len(sentences_data))