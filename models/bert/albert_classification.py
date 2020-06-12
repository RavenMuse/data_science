"""
albert 分类模型
"""
import unicodedata
import re
import keras
import codecs
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.layers import Layer
import keras.backend as K
from keras import initializers, activations

from collections import OrderedDict

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 设置GPU使用率
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

__author__ = "StoneBox"


class BasicTokenizer(object):
    """
    分词器基类
    """

    def __init__(self, do_lower_case=False):
        """
        初始化
        """
        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._do_lower_case = do_lower_case

    def tokenize(self, text, add_cls=True, add_sep=True):
        """
        分词函数
        """
        if self._do_lower_case:
            text = unicodedata.normalize('NFD', text)
            text = ''.join(
                [ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()

        tokens = self._tokenize(text)
        if add_cls:
            tokens.insert(0, self._token_cls)
        if add_sep:
            tokens.append(self._token_sep)
        return tokens

    def token_to_id(self, token):
        """
        token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """
        token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def truncate_sequence(self,
                          max_length,
                          first_sequence,
                          second_sequence=None,
                          pop_index=-1):
        """
        截断总长度
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        """
        输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """
        first_tokens = self.tokenize(first_text)
        if second_text is None:
            second_tokens = None
        else:
            second_tokens = self.tokenize(second_text, add_cls=False)

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] *
                                   (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend(
                    [self._token_pad_id] *
                    (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class Tokenizer(BasicTokenizer):
    """
    Bert原生分词器
    纯Python实现，代码修改自keras_bert的tokenizer实现
    """

    def __init__(self, token_dict, do_lower_case=False):
        """
        初始化
        """
        super(Tokenizer, self).__init__(do_lower_case)
        if isinstance(token_dict, str):
            token_dict = load_vocab(token_dict)

        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        for token in ['pad', 'cls', 'sep', 'unk', 'mask']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass
        self._vocab_size = len(token_dict)

    def token_to_id(self, token):
        """
        token转换为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """
        id转换为对应的token
        """
        return self._token_dict_inv[i]

    def decode(self, ids):
        """
        转为可读文本
        """
        tokens = self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def _tokenize(self, text):
        """
        基本分词函数
        """
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """
        word内分成subword
        """
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

    @staticmethod
    def _is_space(ch):
        """
        空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """
        标点符号类字符判断（全/半角均在此内）
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """
        CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """
        控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """
        判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


class DataGeneratorBase(object):
    """
    数据生成器模版
    """

    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


class DataGenerator(DataGeneratorBase):
    """
    数据生成器
    """

    def __init__(self, data, dict_path, maxlen=180, batch_size=32, is_predict=False):
        self.maxlen = maxlen
        self.dict_path = dict_path
        self.is_predict = is_predict
        super().__init__(data, batch_size)

    def __iter__(self, random=False):

        tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            if not self.is_predict:
                text, label = self.data[i]
            else:
                text = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text, max_length=self.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if not self.is_predict: batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                if not self.is_predict:
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                else:
                    yield [batch_token_ids, batch_segment_ids]
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class PositionEmbedding(Layer):
    """
    定义位置Embedding，这里的Embedding是可训练的
    """

    def __init__(self, input_dim, output_dim, embeddings_initializer, merge_mode='add', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer)

    def call(self, inputs):
        """如果inputs是一个list，则默认第二个输入是位置ids，否则是默认顺序，
        即[0,1,2,3,4,...]"""
        if isinstance(inputs, list):
            inputs, pos_ids = inputs
            pos_embeddings = K.gather(self.embeddings, pos_ids)  # 用pos_ids排序位置向量
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            pos_embeddings = self.embeddings[:seq_len]
            pos_embeddings = K.expand_dims(pos_embeddings, 0)
            pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])
            # 以上操作的作用是:将embedding权重矩阵的shape变换为输入张量的shape
        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的shape
        """
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_moed': self.merge_mode,
            'embeddings_initializer': initializers.serialize(
                self.embeddings_initializer),
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """
    (无Conditional) Layer Normalization
    LN是针对深度网络的某一层的所有神经元的输入进行正态化normalize操作；
    LN中同层的神经元输入具有相同的均值和方差，不同的输入样本有不同的均值和方差
    """

    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = K.epsilon() * K.epsilon()  # keras.backend.epsilon() 1e-07

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

    def call(self, inputs):
        gamma, beta = self.gamma, self.beta
        mean = K.mean(inputs, axis=-1, keepdims=True)  # shape=input_shape
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * gamma + beta
        return outputs


class MultiHeadAttention(Layer):
    """
    多头注意力机制
    """

    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 kernel_initializer='glorot_uniform',
                 max_relative_position=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size  # hidden_size // num_attention_heads
        self.out_dim = heads * head_size  # hidden_size
        self.key_size = key_size if key_size else head_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.max_relative_position = max_relative_position

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.k_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.v_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)
        self.o_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs, q_mask=None, v_mask=None, a_mask=None):
        """
        实现多头注意力
        q_mask:对输入的query序列的mask。
               主要是将输出结果的padding部分置0.
        v_mask:对输入的value序列的mask。
               主要是防止attention读取到padding信息。
        a_mask:对attention矩阵的mask。
               不同的attention mask对应不通的应用。
    例如：
        参数：nb_head=8 ，size_per_head=5 ，x=(bs,6,100)
        权重：WQ\WK\WV=(100,8*5)  ,WO=(8*5,100)
        运算：Q_seq=x*WQ=(bs,6,8*5) reshape 后 (bs,6,8,5) ,转置后 (bs,8,6,5)
        A=Q_seq*K_seq（转置）=(bs,8,6,5)*(bs,8,5,6)=(bs,8,6,6)
        O=A*V_seq=(bs,8,6,6)*(bs,8,6,5)=(bs,8,6,5),转置后 (bs,6,8,5) reshape后 (bs,6,40)
        y=O*WO=(bs,6,40)*(8*5,100)=(bs,6,100)
        """
        # 处理mask
        inputs = inputs[:]
        for i, mask in enumerate([q_mask, v_mask, a_mask]):
            if not mask:
                inputs.insert(3 + i, None)
        q, k, v, q_mask, v_mask = inputs[:5]
        if len(inputs) == 5:
            a_mask = 'history_only'
        elif len(inputs) == 6:
            a_mask = inputs[-1]
        else:
            raise ValueError('wrong inputs for MultiHeadAttention. ')
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        a = a / self.key_size ** 0.5
        a = sequence_masking(a, v_mask, 1, -1)
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'max_relative_position': self.max_relative_position,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """
    FeedForward层，其实就是两个Dense层的叠加
    """

    def __init__(self,
                 units,
                 groups=1,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value

        self.dense_1 = Dense(units=self.units,
                             activation=self.activation,
                             kernel_initializer=self.kernel_initializer)
        self.dense_2 = Dense(units=output_dim,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sequence_padding(inputs, length=None, padding=0):
    """
    Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])
    return outputs


def load_vocab(dict_path):
    """从bert的词典文件中读取词典"""
    token_dict = {}
    with codecs.open(dict_path, encoding='utf-8') as reader:
        i = 0
        for line in reader:
            token = line.strip()
            i += 1
            token_dict[token] = len(token_dict)
    return token_dict


def sequence_masking(x, mask, mode=0, axis=None):
    """
    为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    heads: 相当于batch这一维要被重复的次数。
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis muse be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


def is_one_of(x, ys):
    """
    判断x是否在ys之中
    等价于x in ys，但有些情况下x in ys会报错
    :param x:
    :param ys:
    :return:
    """
    for y in ys:
        if x is y:
            return True
    return False


def piecewise_linear(t, schedule):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if schedule[0][0] != 0:
        schedule = [(0, 0.)] + schedule

    x = K.constant(schedule[0][1], dtype=K.floatx())
    t = K.cast(t, K.floatx())
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1. * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = K.constant(schedule[i][1], dtype=K.floatx())
        x = K.switch(t >= t_begin, x, x_begin)

    return x


def extend_with_piecewise_linear_lr(base_optimizer, name=None):
    """
    返回新的优化器类，加入分段线性学习率
    """

    class new_optimizer(base_optimizer):
        """带有分段线性学习率的优化器
        其中schedule是形如{1000: 1, 2000: 0.1}的字典，
        表示0～1000步内学习率线性地从零增加到100%，然后
        1000～2000步内线性地降到10%，2000步以后保持10%
        """

        def __init__(self, lr_schedule, *args, **kwargs):
            super(new_optimizer, self).__init__(*args, **kwargs)
            self.lr_schedule = {int(i): j for i, j in lr_schedule.items()}

        @K.symbolic
        def get_updates(self, loss, params):
            lr_multiplier = piecewise_linear(self.iterations, self.lr_schedule)

            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params):
                    new_x = x + (new_x - x) * lr_multiplier
                return old_update(x, new_x)

            K.update = new_update
            updates = super(new_optimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def get_config(self):
            config = {'lr_schedule': self.lr_schedule}
            base_config = super(new_optimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    if isinstance(name, str):
        new_optimizer.__name__ = name
        keras.utils.get_custom_objects()[name] = new_optimizer

    return new_optimizer


class AlbertClassification:

    def __init__(self, config_path, checkpoint_path, dict_path, class_num, max_length=256):
        """
        albert模型参数初始化
        :param config_path: 配置文件路径 albert_tiny_zh_google/albert_config_tiny_g.json
        :param checkpoint_path:  预训练模型路径 albert_tiny_zh_google/albert_model.ckpt
        :param dict_path: 词典路径 albert_tiny_zh_google/vocab.txt
        :param class_num:  类数
        :param max_length: 序列最大长度
        """
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path
        config = json.load(open(config_path))
        self.vocab_size = config['vocab_size']
        self.max_position_embeddings = config.get('max_position_embeddings')
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.intermediate_size = config['intermediate_size']
        self.hidden_act = config['hidden_act']
        self.dropout_rate = config['hidden_dropout_prob']
        self.initializer_range = config.get('initializer_range')
        self.embedding_size = config.get('embedding_size')
        self.with_pool = True
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.model_initializer = keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.max_relative_position = None
        self.with_nsp = False
        self.with_mlm = False
        self.class_num = class_num
        self.max_length = max_length
        self.model = None

    def transformer_block(self, inputs, sequence_mask, attention_mask=None, attention_name='attention',
                          feed_forward_name='feed-forward', layers=None):
        """
        构建单个的Transformer Block
        如果没有传入layers则新建层；如果传入则重用旧层
        """
        x = inputs
        layers = layers if layers is not None else [
            MultiHeadAttention(heads=self.num_attention_heads,
                               head_size=self.attention_head_size,
                               kernel_initializer=self.model_initializer,
                               name=attention_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropouintermediate_sizet' % attention_name),
            Add(name='%s-Add' % attention_name),
            LayerNormalization(name='%s-Norm' % attention_name),
            FeedForward(units=self.intermediate_size,
                        groups=1,
                        activation=self.hidden_act,
                        kernel_initializer=self.model_initializer,
                        name=feed_forward_name),
            Dropout(rate=self.dropout_rate,
                    name='%s-Dropout' % feed_forward_name),
            Add(name='%s-Add' % feed_forward_name),
            LayerNormalization(name='%s-Norm' % feed_forward_name),
        ]
        xi = x
        if attention_mask is None:
            x = layers[0]([x, x, x, sequence_mask], v_mask=True)
        elif attention_mask is 'history_only':
            x = layers[0]([x, x, x, sequence_mask], v_mask=True, a_mask=True)
        else:
            x = layers[0]([x, x, x, sequence_mask, attention_mask],
                          v_mask=True, a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](x)
        # Feed Forward
        xi = x
        x = layers[4](x)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](x)
        return x, layers

    def variable_mapping(self, variable_names, model):
        """
        构建Keras层与checkpoint的变量名之间的映射表
        :param variable_names:
        :param model:
        :return:
        """
        mapping = OrderedDict()

        mapping['Embedding-Token'] = ['bert/embeddings/word_embeddings']
        mapping['Embedding-Segment'] = ['bert/embeddings/token_type_embeddings']
        if self.max_relative_position is None:
            mapping['Embedding-Position'] = ['bert/embeddings/position_embeddings']

        mapping['Embedding-Norm'] = [
            'bert/embeddings/LayerNorm/gamma',
            'bert/embeddings/LayerNorm/beta',
        ]
        if self.embedding_size != self.hidden_size:
            mapping['Embedding-Mapping'] = [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ]

        for i in range(self.num_hidden_layers):
            try:
                model.get_layer('Encoder-%d-MultiHeadSelfAttention' % (i + 1))
            except ValueError:
                continue
            if ('bert/encoder/layer_%d/attention/self/query/kernel' % i) in variable_names:
                block_name = 'layer_%d' % i
            else:
                block_name = 'transformer/group_0/inner_group_0'

            mapping['Encoder-%d-MultiHeadSelfAttention' % (i + 1)] = [
                'bert/encoder/%s/attention/self/query/kernel' % block_name,
                'bert/encoder/%s/attention/self/query/bias' % block_name,
                'bert/encoder/%s/attention/self/key/kernel' % block_name,
                'bert/encoder/%s/attention/self/key/bias' % block_name,
                'bert/encoder/%s/attention/self/value/kernel' % block_name,
                'bert/encoder/%s/attention/self/value/bias' % block_name,
                'bert/encoder/%s/attention/output/dense/kernel' % block_name,
                'bert/encoder/%s/attention/output/dense/bias' % block_name,
            ]
            mapping['Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)] = [
                'bert/encoder/%s/attention/output/LayerNorm/gamma' % block_name,
                'bert/encoder/%s/attention/output/LayerNorm/beta' % block_name,
            ]
            mapping['Encoder-%d-FeedForward' % (i + 1)] = [
                'bert/encoder/%s/intermediate/dense/kernel' % block_name,
                'bert/encoder/%s/intermediate/dense/bias' % block_name,
                'bert/encoder/%s/output/dense/kernel' % block_name,
                'bert/encoder/%s/output/dense/bias' % block_name,
            ]
            mapping['Encoder-%d-FeedForward-Norm' % (i + 1)] = [
                'bert/encoder/%s/output/LayerNorm/gamma' % block_name,
                'bert/encoder/%s/output/LayerNorm/beta' % block_name,
            ]

        if self.with_pool or self.with_nsp:
            mapping['Pooler-Dense'] = [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ]
            if self.with_nsp:
                mapping['NSP-Proba'] = [
                    'cls/seq_relationship/output_weights',
                    'cls/seq_relationship/output_bias',
                ]

        if self.with_mlm:
            mapping['MLM-Dense'] = [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ]
            mapping['MLM-Norm'] = [
                'cls/predictions/transform/LayerNorm/gamma',
                'cls/predictions/transform/LayerNorm/beta',
            ]
            mapping['MLM-Proba'] = ['cls/predictions/output_bias']

        return mapping

    def load_variable(self, name, variable_names):
        """
        加载单个变量的函数
        :param name:
        :param variable_names:
        :return:
        """
        sims = [similarity(name, n) for n in variable_names]
        found_name = variable_names.pop(np.argmax(sims))
        #     print('==> searching: %s, found name: %s' % (name, found_name))
        variable = tf.train.load_variable(self.checkpoint_path, found_name)
        if name in [
            'bert/embeddings/word_embeddings',
            'cls/predictions/output_bias',
        ]:
            return variable
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def load_variables(self, names, variable_names):
        """
        批量加载的函数
        :param names:
        :param variable_names:
        :return:
        """
        if not isinstance(names, list):
            names = [names]
        return [self.load_variable(name, variable_names) for name in names]

    def set_gelu(self, version):
        """
        设置gelu版本
        """

        def gelu_erf(x):
            """
            基于Erf直接计算的gelu函数
            """
            return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))

        def gelu_tanh(x):
            """
            基于Tanh近似计算的gelu函数
            """
            cdf = 0.5 * (1.0 + K.tanh(
                (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
            return x * cdf

        version = version.lower()
        assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
        if version == 'erf':
            keras.utils.get_custom_objects()['gelu'] = gelu_erf
        else:
            keras.utils.get_custom_objects()['gelu'] = gelu_tanh

    def build(self):
        # 构建输入层,x_in为句子token输入，s_in是segment_id，用于区分第一个和第二个句子，
        # 这里只有第一个句子，所以segment_id都为0
        # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
        # (a) 两个句子:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) 单个句子:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        # 这里 "type_ids" 主要用于区分第一个第二个句子。
        # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
        # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

        self.set_gelu('tanh')
        x_in = Input(shape=(None,), name='Input-Token')
        s_in = Input(shape=(None,), name='Input-Segment')
        x, s = input_layers = [x_in, s_in]

        # 自行构建Mask
        sequence_mask = Lambda(lambda x: K.cast(K.greater(x, 0), K.floatx()),
                               name='Sequence-Mask')(x)

        # Embedding部分
        # 输入Embedding，包括输入
        x = Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.model_initializer,
                      name='Embedding-Token')(x)
        s = Embedding(input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.model_initializer,
                      name='Embedding-Segment')(s)

        x = Add(name='Embedding-Token-Segment')([x, s])
        # 默认的权重初始化方法：截断正态分布
        x = PositionEmbedding(input_dim=self.max_position_embeddings,
                              output_dim=self.embedding_size,
                              merge_mode='add',
                              embeddings_initializer=self.model_initializer,
                              name='Embedding-Position')(x)
        x = LayerNormalization(name='Embedding-Norm')(x)

        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        if self.embedding_size != self.hidden_size:
            x = Dense(units=self.hidden_size,
                      kernel_initializer=self.model_initializer,
                      name='Embedding-Mapping')(x)
        # 主要transformer部分
        layers = None
        for i in range(self.num_hidden_layers):
            attention_name = 'Encoder-%d-MultiHeadSelfAttention' % (i + 1)
            feed_forward_name = 'Encoder-%d-FeedForward' % (i + 1)
            x, layers = self.transformer_block(inputs=x,
                                               sequence_mask=sequence_mask,
                                               attention_mask=None,
                                               attention_name=attention_name,
                                               feed_forward_name=feed_forward_name,
                                               layers=layers)  # albert中参数共享，transform块重复
        outputs = [x]
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = Lambda(lambda x: x[:, 0], name='Pooler')(x)
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = Dense(units=self.hidden_size,
                      activation=pool_activation,
                      kernel_initializer=self.model_initializer,
                      name='Pooler-Dense')(x)
            outputs.append(x)
        outputs += []
        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        # 模型构建完成
        albert_model = keras.models.Model(input_layers, outputs)

        # 从预训练好的Bert的checkpoint中加载权重
        # 为了简化写法，对变量名的匹配引入了一定的模糊匹配能力。
        variable_names = [
            n[0] for n in tf.train.list_variables(self.checkpoint_path)
            if 'adam' not in n[0]
        ]
        mapping = self.variable_mapping(variable_names, albert_model)

        for layer_name, layer_variable_names in mapping.items():
            values = self.load_variables(layer_variable_names, variable_names)
            weights = albert_model.get_layer(layer_name).trainable_weights
            if 'Norm' in layer_name:
                weights = weights[:2]
            if len(weights) != len(values):
                raise ValueError(
                    'Expecting %s weights, but provide a list of %s weights.'
                    % (len(weights), len(values))
                )
            K.batch_set_value(zip(weights, values))

        # 拼接末尾分类层（softmax）
        out_put = Dropout(rate=0.1)(albert_model.output)
        out_put = Dense(units=self.class_num,
                        activation='softmax',
                        kernel_initializer=self.model_initializer)(out_put)
        self.model = keras.models.Model(albert_model.input, out_put)

        Adam = keras.optimizers.Adam
        AdamLR = extend_with_piecewise_linear_lr(Adam)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            # optimizer=Adam(1e-5),  # 用足够小的学习率
            optimizer=AdamLR(learning_rate=1e-4,
                             lr_schedule={1000: 1, 2000: 0.1}),
            metrics=['accuracy'],
        )
        return self.model

    def train(self, data, validation_proportion=0.4, batch_size=32, epochs=10):
        """
        模型训练
        :param data:
        :param validation_proportion:
        :param batch_size:
        :param epochs:
        :return:
        """

        if not self.model:
            print("please build model before train")
            return

        train_sample, validation_sample = train_test_split(data, test_size=validation_proportion, shuffle=True)

        train_generator = DataGenerator(train_sample, self.dict_path, self.max_length, batch_size)
        validation_generator = DataGenerator(validation_sample, self.dict_path, self.max_length,
                                             batch_size=len(validation_sample))

        class Evaluator(keras.callbacks.Callback):

            def __init__(self, model, validation_generator):
                self.best_val_acc = 0.
                self.validation_generator = validation_generator
                self.model = model

            def on_epoch_end(self, epoch, logs=None):
                validation_x = list(map(lambda x: x[0], self.validation_generator))
                validation_y = list(map(lambda x: x[1][:, 0], self.validation_generator))[0]
                prediction = self.model.predict(list(validation_x)[0]).argmax(axis=1)
                print(classification_report(validation_y, prediction))
                val_acc = accuracy_score(validation_y, prediction)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.model.save_weights('albert_classification_best_model.weights')

        self.model.fit_generator(train_generator.forfit(),
                                 steps_per_epoch=len(train_generator),
                                 epochs=epochs,
                                 callbacks=[Evaluator(self.model, validation_generator)])

    def predict(self, data, batch_size=32):
        self.model.load_weights('albert_classification_best_model.weights')

        test_generator = DataGenerator(data, self.dict_path, self.max_length, batch_size, is_predict=True)
        for bs in test_generator:
            yield self.model.predict(bs)


if __name__ == '__main__':
    # 测试
    # 预训练模型：https://github.com/google-research/bert
    config_path = 'albert_tiny_zh_google/albert_config_tiny_g.json'
    checkpoint_path = 'albert_tiny_zh_google/albert_model.ckpt'
    dict_path = 'albert_tiny_zh_google/vocab.txt'

    train_data = pd.read_csv("train_data_sample.csv")
    albert_classifier = AlbertClassification(config_path, checkpoint_path, dict_path, class_num=2, max_length=256)
    albert_classifier.build()
    albert_classifier.train(train_data[['text', 'label']].values,
                            validation_proportion=0.4,
                            batch_size=32,
                            epochs=10)
    data = pd.read_csv("test_data_sample.csv")
    predict_iter = albert_classifier.predict(train_data['text'].values, batch_size=32)
    for batch in predict_iter:
        print(batch)  # [[0.4,0.6],[0.7,0.3],[0.6,0.4]]
        print(batch.argmax(axis=1))  # [1,0,0]
        break
