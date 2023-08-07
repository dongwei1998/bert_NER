# coding=utf-8
# =============================================
# @Time      : 2022-04-06 17:08
# @Author    : DongWei1998
# @FileName  : data_help.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
from utils import parameter
from utils import tokenization
import tensorflow as tf


# 数据读取 【test label】
def _read_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        words = []
        labels = []
        label_n = 4
        label_2_id = {"X":1, "[CLS]":2, "[SEP]":3}
        for line in f:
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[1])
                if tokens[1] not in label_2_id.keys():
                    label_2_id[tokens[1]] = label_n
                    label_n += 1
            else:
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
        return lines,label_2_id



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask

def _create_example(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text = tokenization.convert_to_unicode(line[0])
        label = tokenization.convert_to_unicode(line[1])
        examples.append(InputExample(guid=guid, text=text, label=label))
    return examples

# 文本序列化
def filed_based_convert_examples_to_features(examples, label_2_id, max_seq_length, tokenizer):
    label_data = []
    data  =  []
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        textlist = example.text.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:  # 一般不会出现else
                    labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")  # 句子开始设置CLS 标志
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_2_id["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_2_id[labels[i]])
        ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_2_id["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
        input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        # padding, 使用
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        data.append(input_ids)
        label_data.append(label_ids)
    return  data ,label_data

def data_set(args,file,set_type):
    # 获取bert词表
    vocab_file = args.vocab_file
    input_file= file
    # 文本序列化工具
    tokenizer= tokenization.FullTokenizer(vocab_file=vocab_file)
    # 读取数据
    lines,label_2_id = _read_data(input_file)
    # label_map 持久化
    if os.path.exists(args.label_2_id_dir):
        with open(args.label_2_id_dir,'r',encoding='utf-8') as r:
            label_2_id = json.loads(r.read())
    else:
        with open(args.label_2_id_dir,'w',encoding='utf-8') as w:
            w.write(json.dumps(label_2_id))

    # 构建序列对象
    examples = _create_example(lines, set_type)
    # 文本数据序列化
    data, label = filed_based_convert_examples_to_features(examples, label_2_id, args.max_seq_length, tokenizer)
    # 将数据转换为dataset格式
    train_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    #对数据进行打乱 批次话 buffer_size 总数据量   batch 批次大小
    train_dataset = train_dataset.shuffle(buffer_size=len(data)).batch(args.batch_size)

    # # 参数更新
    # with open(args.vocab_file,'r',encoding='utf-8') as r:
    #     input_vocab_size = r.readlines()
    # args.input_vocab_size = len(input_vocab_size)
    # args.num_calss = len(label_2_id)

    return train_dataset,args


def alphamind_read_data(text_data_file,label_data_file):
    lines = []
    label_2_id = {"X": 1, "[CLS]": 2, "[SEP]": 3}
    label_n = 4
    text_obj = open(text_data_file,'r',encoding='utf-8')
    label_obj = open(label_data_file, 'r', encoding='utf-8')
    text_data = text_obj.readlines()
    label_data = label_obj.readlines()
    if len(text_data) == len(label_data):
        for i in range(len(text_data)):
            words = text_data[i].split(' ')
            labels = label_data[i].split(' ')
            if len(words) == len(labels):
                l = []
                w = ' '.join([word for word in words if len(word) > 0])
                for label in labels :
                    if label not in label_2_id.keys():
                        label_2_id[label] = label_n
                        label_n += 1
                    if len(label) > 0:
                        l.append(label)
                l = ' '.join(l)
                lines.append([w, l])
            else:
                raise args.logger.info(f"text_data{i} len is {text_data[i]} --- label_data{i} len is {label_data[i]}")
    else:
        raise args.logger.info(f'text_data len is {len(text_data)}  --- label_data len is {len(label_data)} ')
    text_obj.close()
    label_obj.close()
    return lines, label_2_id



def data_set_alphamind(args,set_type,mode):
    if mode == 'train':
        text_data_file = os.path.join(args.data_dir,'text_train.txt')
        label_data_file = os.path.join(args.data_dir, 'labels_train.txt')
    elif mode == 'dev':
        text_data_file = os.path.join(args.data_dir, 'text_val.txt')
        label_data_file = os.path.join(args.data_dir, 'labels_val.txt')
    elif mode == 'test':
        text_data_file = os.path.join(args.data_dir, 'text_test.txt')
        label_data_file = os.path.join(args.data_dir, 'labels_test.txt')
    else:
        raise args.logger.info('mode value is not in [train dev test] ')

    # 文本序列化工具
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file)
    # 读取数据
    lines, label_2_id = alphamind_read_data(text_data_file,label_data_file)
    # label_map 持久化
    if os.path.exists(args.label_2_id_dir):
        with open(args.label_2_id_dir, 'r', encoding='utf-8') as r:
            label_2_id = json.loads(r.read())
    else:
        with open(args.label_2_id_dir, 'w', encoding='utf-8') as w:
            w.write(json.dumps(label_2_id))
    # 构建序列对象
    examples = _create_example(lines, set_type)
    # 文本数据序列化
    data, label = filed_based_convert_examples_to_features(examples, label_2_id, args.max_seq_length, tokenizer)
    # 将数据转换为dataset格式
    train_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    # 对数据进行打乱 批次话 buffer_size 总数据量   batch 批次大小
    # fixme
    train_dataset = train_dataset.shuffle(buffer_size=len(data)).batch(args.batch_size)

    # # 参数更新
    # with open(args.vocab_file,'r',encoding='utf-8') as r:
    #     input_vocab_size = r.readlines()
    # args.input_vocab_size = len(input_vocab_size)
    # args.num_calss = len(label_2_id)

    return train_dataset, args




if __name__ == '__main__':
    alphamind_read_data(text_data_file='../datasets/text_val.txt', label_data_file='../datasets/labels_val.txt')

    args = parameter.parser_opt('train')

    train_dataset = data_set(args,file='../datasets/example.dev',set_type=True)

    # # 数据可视化
    # total = 0
    # for times,item in enumerate(train_dataset,1):
    #     # 遍历训练数据，相当于一个epoch
    #     if times < 20 :
    #         print(f'=======当前批数：{times} ========')
    #         print(item)
    #         print(item[0].shape)
    #     batch_count = item[0].shape[0]   # batch_size设置为512,但是不一定能整除,实际最后一个batch达不到512
    #     total += batch_count
    #     times += 1
    #
    # print('扫过数据数量:', total)
