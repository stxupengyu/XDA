import os
from tqdm import tqdm

def get_train_examples(args):
    train_texts = read_txt(os.path.join(args.data_dir, args.da_train_texts), args)
    train_labels = read_txt(os.path.join(args.data_dir, args.da_train_labels), args)
    return train_texts, train_labels

def read_txt(txt_path, args):
    f = open(txt_path, 'r')
    lines = []
    count = 0
    for line in f.readlines():
        line = line.strip()[:args.cut_length]
        count += 1
        if count > args.sample_num:
            break
        # process src and trg line
        lines.append(line)
    return lines

def save_the_new(new_docs, new_labels, args):
    save_train_path_label = os.path.join(args.output_dir, args.filtered_train_labels)
    save_train_file_label = open(save_train_path_label, 'w')
    with save_train_file_label as flabel:
        for new_label in tqdm(new_labels):
            for inst in new_label:
                flabel.write(inst + '\n')
    print('label augmentation completed')

    save_train_path = os.path.join(args.output_dir, args.filtered_train_texts)
    save_train_file = open(save_train_path, 'w')
    with save_train_file as f:
        for new_doc in new_docs:
            for inst in new_doc:
                # print(new_docs)
                # print(new_doc)
                # print(inst)
                f.write(inst[0] + '\n')
    print('text augmentation completed')
    print('dataset size = %s' % len(new_docs))
    print("generated augmented sentences with bert " + " with num_aug=" + str(args.aug_num))