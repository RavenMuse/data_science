# -*- encoding:utf-8 -*-
import os
import csv
import json
import h5py
import itertools
import numpy as np
import pandas as pd
import plotly
import operator

from datetime import *
# from pandasql import sqldf
from PIL import Image
from plotly.graph_objs import *
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, accuracy_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt

__author__ = "StoneBox"

# plotly.tools.set_credentials_file(username='stone_box', api_key='gtetznvbrbHxbz76ckZZ')

# pysql = lambda q: sqldf(q, globals())
# pysql = sqldf

# 设置最大info显示行数
pd.set_option('display.max_info_columns', 1000)
pd.set_option('display.max_columns', 1000)


def make_dir(path):
    """
    创建文件夹
    :param path:  全路径 ,例如 "/home/raven/test_dir" 则创建test_dir文件夹
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        # "%s is %s"%("12","12")
        print("%s is exist" % path)


def list_file(path, filter_str=None):
    """
    显示路径下的所有文件及文件夹
    :param filter_str: 过滤包含字符串的文件; 格式： string
    :param path:
    :return:
    """

    if filter_str is None:
        result = os.listdir(path)
    else:
        result = list(filter(lambda x: x.__contains__(filter_str), os.listdir(path)))
    print("==============Path: %s List==============" % path)
    print(result)
    print("=============================================")
    return result


def reverse_dict(cur_dict):
    """
    反转字典key和value
    :param cur_dict:
    :return:
    """
    return dict([(value, key) for (key, value) in cur_dict.items()])


def read_json(json_str):
    """
    读取json字符串,转化为dict对象
    :param json_str:
    :return: dict
    """
    return json.loads(json_str)


def to_json(dic):
    """
    将dict转化为json字符串
    :param dic:
    :return: json字符串
    """
    return json.dumps(dic)


def printf(msg):
    """
    时间戳打印
    :param msg:
    :return:
    """
    strtowrite = "[{}] {}".format(datetime.now(), msg)
    print(strtowrite)


def sort_dict_by_value(dic, desc):
    """
    字典排序(按值)
    :param dic:
    :param desc: 是否降序
    :return:
    """
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=desc)
    return sorted_x


def sort_dict_by_key(dic, desc):
    """
    字典排序(按键)
    :param dic:
    :param desc: 是否降序
    :return:
    """
    sorted_x = sorted(dic.items(), key=operator.itemgetter(0), reverse=desc)
    return sorted_x


class IOTools:
    """
    数据读写工具
    包括: read_csv;write_numpy_object;read_numpy_object;write_hdf5;write_hdf5_by_groups;read_hdf5
    """

    @staticmethod
    def read_csv(path, has_header=True):
        """
        读csv文件
        :param path:
        :param has_header:
        :return: (data[[value1,value2,...]],list(columns_name))
        """
        try:
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader) if has_header else ""
                data = [row for row in reader]
        except csv.Error as e:
            print("Error rreading CSV file as line %s: %s" % (reader.line_num, e))
            sys.exit(-1)
        if has_header:
            print("========= columns name =============")
            print(header)
            print("====================================")

        return data, header

    @staticmethod
    def write_numpy_object(object, path):
        """
        保存numpy object
        :param object:numpy object,例如:np.array
        :param path:
        :return:
        """
        np.save(path, object)

    @staticmethod
    def read_numpy_object(path):
        """
        载入numpy object
        :param path:
        :return:
        """
        np.load(path)

    @staticmethod
    def write_hdf5(data, path):
        """
        将数据写入 hdf5 (.h5) 文件
        :param data: 格式: dict(dataset_name->data) , 例如: {'label':label_data,'data':data}
        :param path:
        :return:
        """
        with h5py.File(path, 'w') as f:
            if isinstance(data, dict):
                for key in data:
                    f.create_dataset(key, data=data[key])
            else:
                raise TypeError("Type of data should be dict.")

    @staticmethod
    def write_hdf5_by_groups(data, groups, path):
        """
        将数据写入 hdf5 (.h5) 文件
        :param data:  格式: list(dict(dataset_name->data)) , 例如: [{'sample1':data,'sample2':data},{'sample3':data,'sample4':data}]
        :param groups: 格式: list(group) , 例如: ['group1','group2']
        :param path:
        :return: 数据集
        """

        with h5py.File(path, 'w') as f:
            if not isinstance(groups, list): raise TypeError("Type of groups should be list.")

            if isinstance(data, list) and isinstance(data[0], dict):
                for i in range(0, len(groups)):
                    group = f.require_group(groups[i])
                    for key in data[i]: group.create_dataset(key, data=data[i][key])
            else:
                raise TypeError("Type of data should be list(dict).")

    @staticmethod
    def read_hdf5(path, group=None, dataset_name=None):
        """
        从 hdf5 (.h5) 文件读取数据
        :param path:
        :param group: 数据组名
        :param dataset_name: 数据集名
        :return: 数据集
        """
        result = None
        with h5py.File(path, 'r') as f:
            if group is not None:
                if dataset_name is not None:
                    result = f[group][dataset_name].value
                else:
                    raise ValueError("dataset_name should not be None.")
            else:
                if dataset_name is not None:
                    result = f[dataset_name].value
                else:
                    raise ValueError("dataset_name should not be None.")
        return result


class ImageTools:
    """
    图像处理工具(pillow)
    包括: ImageSlicer;ImageCombiner;read_image;to_np_array;to_gray_image;to_pil_image;get_histogram;save_image
    扩展链接:http://pillow.readthedocs.io/en/latest/
    """
    IPYTHON_MODE = False

    class ImageSlicer:
        """
        根据切分图片大小切分指定图片,返回一个迭代器,迭代切分图片和排版位置
        """

        def __init__(self, pil_image, box_size):
            """

            :param pil_image:
            :param box_size: (宽,高)
            """
            self._image = pil_image
            xsize, ysize = pil_image.size
            self._width = box_size[0]
            self._height = box_size[1]
            self._x_num = int(xsize / self._width)
            self._y_num = int(ysize / self._height)
            self._iter = itertools.product(range(self._y_num), range(self._x_num))

        def __iter__(self):
            return self

        def __len__(self):
            return self._x_num * self._y_num

        def __next__(self):
            """

            :return: 切分图片
            """
            (y, x) = self._iter.__next__()
            # crop (a,b,c,d) , (a,b) left top position , (c,d) right bottom position
            return self._image.crop(
                (x * self._width, y * self._height, (x + 1) * self._width, (y + 1) * self._height)), (y, x)

    class ImageCombiner:
        """
        通过切分图片合成一张图片
        """

        def __init__(self, size, mode="RGB"):
            """
            初始化生成图片
            :param size: 生成图片的大小, 格式:(宽,高)
            :param mode: 图片格式
            """
            self._new_image = Image.new(mode, size)

        def append(self, pil_image, positon):
            """
            按排版位置添加切片图片
            :param pil_image:
            :return:
            """
            width, height = pil_image.size
            self._new_image.paste(pil_image, (positon[1] * width, positon[0] * height,
                                              (positon[1] + 1) * width, (positon[0] + 1) * height))

        def finish(self):
            """
            返回生成的图片
            :return:
            """
            return self._new_image

    @staticmethod
    def read_image(path):
        """
        读取图片
        :param path:
        :return: pil图像
        """
        im = Image.open(path)

        print("========== Image Info ==========")
        print(" format: {0} \n width/height: {1} \n mode: {2}".format(im.format, im.size, im.mode))
        print("================================")
        return im

    @staticmethod
    def to_np_array(pil_image):
        """
        pil图像转化为numpy array
        :param pil_image:
        :return: (width,height,channel) with value from 0 - 255
        """
        return np.array(pil_image)

    @staticmethod
    def to_gray_image(pil_image):
        """
        生成pil图像灰度图
        灰度值公式：L = R * 299/1000 + G * 587/1000 + B * 114/1000
        :param pil_image:
        :return: 灰度图,其中值范围为[0,255]
        """
        return pil_image.convert("L")

    @staticmethod
    def to_pil_image(array):
        """
        numpy array转化为pil图像
        :param pil_image:
        :return:
        """
        return Image.fromarray(array.round().astype(np.uint8))

    @staticmethod
    def show(pil_image):
        """
        显示图片,IPython模式下显示在页面中,否则打开图片显示
        :param pil_image:
        :return:
        """
        if ImageTools.IPYTHON_MODE:
            plt.imshow(pil_image)
            plt.show()
        else:
            pil_image.show()

    @staticmethod
    def get_histogram(pil_image):
        """
        统计RGB值的像素数
        :param pil_image:
        :return: 返回list,RGB图像list长度为256*3=768 ,灰度图为256
        """
        return pil_image.histogram()

    @staticmethod
    def save_image(pil_image, path, file_type="png"):
        """
        保存pil图像
        :param pil_image:
        :param path: 文件路径
        :param file_type: 图片格式 'png' 或 'jpeg'
        :return:
        """
        pil_image.save(path, file_type)


class PandasTools:
    """
    Pandas工具
    包括: read_csv;read_big_csv;write_csv;show_sample;join;read_hdf5;
    """

    @staticmethod
    def read_csv(path, schema=None, sep=',', has_header=True):
        """
        读取csv文件
        :param path: 文件路径
        :param schema:表结构
        :param sep: csv列分割符
        :param has_header: 是否有标题
        :return: 返回dataframe读取器;
                 格式:dataframe
        """
        dataframe = pd.read_csv(path, header=0 if has_header else None, sep=sep, dtype=schema)

        if has_header:
            print("============= Schema Info ===============")
            print("Path:%s" % path)
            print(dataframe.info())
            print("=========================================")

        return dataframe

    @staticmethod
    def read_big_csv(path, schema=None, sep=',', has_header=True):
        """
        读取csv大文件
        :param path: 文件路径
        :param schema:表结构
        :param sep: csv列分割符
        :param has_header: 是否有表头
        :return: 返回reader可以通过reader.get_chunk(num)方法获取下一段数量的dataframe;
                 格式:reader
        """

        temp = pd.read_csv(path, header=0 if has_header else None, sep=sep, iterator=True, dtype=schema)

        reader = pd.read_csv(path, header=0 if has_header else None, sep=sep, iterator=True, dtype=schema)

        if has_header:
            print("============= Schema Info ===============")
            print("Path:%s" % path)
            print(temp.get_chunk(1).dtypes)
            print("=========================================")
            temp.close()

        return reader

    @staticmethod
    def write_csv(data_frame, path, sep=',', has_header=True, has_index=False):
        """
        写csv文件
        :param data_frame:
        :param path: 文件路径
        :param sep: csv列分割符
        :param has_header: 是否包含表头
        :param has_index: 是否包含索引
        :return:
        """
        data_frame.to_csv(path, index=has_index, header=has_header, sep=sep, encoding="utf-8")

    @staticmethod
    def show_sample(data_frame):
        """

        :param data_frame:
        :return: None
        """
        print(data_frame.head(10))

    @staticmethod
    def join(data_frame_1, data_frame_2, on, how="inner"):
        """

        :param data_frame_1:
        :param data_frame_2:
        :param on: list(key) , such as [key1,key2]
        :param how: left,right,outer
        :return: dataframe
        """
        return pd.merge(data_frame_1, data_frame_2, on=on, how=how)


class ChartTools:
    """
    图表工具(plotly)
    包括: plot_scatter;plot_area;plot_box:plot_scatter3d;plot_surface3d;plot_bar;plot_pie;plot_heat_map
    扩展链接:https://plot.ly/python/#fundamentals
    """
    IPYTHON_MODE = False

    @staticmethod
    def _plot_fig(data, title="", is_image=False, is_online=False, **other_layout):

        other_layout['title'] = title

        if is_online:
            plotly.plotly.plot({
                "data": data,
                "layout": Layout(other_layout)
            }, filename=title, auto_open=False, fileopt='overwrite')
        else:
            plotly.offline.init_notebook_mode()
            ploter = plotly.offline.iplot if ChartTools.IPYTHON_MODE else plotly.offline.plot
            ploter({
                "data": data,
                "layout": Layout(other_layout)
            }, filename=title + '.html', image='png' if is_image else None)

    @staticmethod
    def plot_scatter(data, traces, title="", mode="lines+markers", is_image=False, is_online=False):
        """
        plot scatter
        :param data: [(x,y),(x2,y2)], such as the data of two trace : [([1,2,3],[10,20,30]),([4,5,6],[14,15,18])]
        :param traces: ['trace'],the name of traces ,such as ['trace1','trace2']
        :param title: title of current chat
        :param mode: "lines", "markers", "lines+markers", "lines+markers+text", "none"
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [Scatter(x=item[1][0], y=item[1][1], name=item[0], mode=mode) for item in zip(traces, data)]

        ChartTools._plot_fig(data, title, is_image, is_online)

    @staticmethod
    def plot_area(data, traces, title="", is_image=False, is_online=False):
        """
        plot scatter
        :param data: [(x,y),(x2,y2)], such as the data of two trace : [([1,2,3],[10,20,30]),([4,5,6],[14,15,18])]
        :param traces: ['trace'],the name of traces ,such as ['trace1','trace2']
        :param title: title of current chat
        :param is_image: show html with download tip when true
        :param is_online:
        :return:

        """

        data = [Scatter(x=item[1][0], y=item[1][1], name=item[0], line=dict(width=0.5), fill='tozeroy') for item in
                zip(traces, data)]

        ChartTools._plot_fig(data, title, is_image, is_online)

    @staticmethod
    def plot_box(data, traces, title="", is_image=False, is_online=False):
        """
        plot box
        :param data: [y1,y2], such as the data of two traces : [[1,2,3],[10,20,30]]
        :param traces: ['trace'],the name of traces ,such as ['trace1','trace2']
        :param title: title of current chat
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [Box(y=item[1], name=item[0], boxpoints='all', jitter=0.3, pointpos=-1.8)
                for item in zip(traces, data)]

        ChartTools._plot_fig(data, title, is_image, is_online)

    @staticmethod
    def plot_scatter3d(data, traces, title="", mode="markers", is_image=False, is_online=False):
        """
        plot scatter
        :param data: [(x,y,z),(x2,y2,z2)], such as the data of two trace : [([1,2,3],[10,20,30],[12,13,14]),([4,5,6],[14,15,18],[12,13,14])]
        :param traces: ['trace'],the name of traces ,such as ['trace1','trace2']
        :param title: title of current chat
        :param mode: "lines", "markers", "lines+markers", "lines+markers+text", "none"
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [Scatter3d(
            x=item[1][0],
            y=item[1][1],
            z=item[1][2],
            name=item[0],
            mode=mode,
            marker=dict(
                size=2,
                line=dict(
                    width=0.3
                ),
                opacity=0.8,

            )) for item in zip(traces, data)]

        ChartTools._plot_fig(data, title, is_image, is_online)

    @staticmethod
    def plot_surface3d(data, traces, title="", mode="markers", is_image=False, is_online=False):
        """
        plot scatter
        :param data: [(x,y,z),(x2,y2,z2)], such as the data of two trace : [([1,2,3],[10,20,30],[12,13,14]),([4,5,6],[14,15,18],[12,13,14])]
        :param traces: ['trace'],the name of traces ,such as ['trace1','trace2']
        :param title: title of current chat
        :param mode: "lines", "markers", "lines+markers", "lines+markers+text", "none"
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [Scatter3d(
            x=item[1][0],
            y=item[1][1],
            z=item[1][2],
            name=item[0],
            mode=mode,
            marker=dict(
                size=2,
                line=dict(
                    width=0.3
                ),
                opacity=0.8,

            )) for item in zip(traces, data)]

        ChartTools._plot_fig(data, title, is_image, is_online)

    @staticmethod
    def plot_bar(data, traces, title="", orientation="v", barmode="group", is_image=False, is_online=False):
        """
        plot scatter
        :param data: [(x,y),(x2,y2)], such as the data of two trace : [(['a','b','c'],[10,20,30]),(['a','b','c'],[14,15,18])]
        :param traces: ['trace'],the name of traces ,such as ['trace1','trace2']
        :param title: title of current chat
        :param orientation: 'v' or 'h'
        :param barmode: 'stack' or 'group'
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [
            Bar(x=item[1][0], y=item[1][1], name=item[0], text=item[1][1], textposition='auto', orientation=orientation)
            for item in
            zip(traces, data)]

        ChartTools._plot_fig(data, title, is_image, is_online, barmode=barmode)

    @staticmethod
    def plot_pie(data, title="", is_image=False, is_online=False):
        """
        plot scatter
        :param data: (label,values), example : (['a','b','c'],[10,20,30])
        :param title: title of current chat
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [Pie(labels=data[0], values=data[1], hole=.4)]

        ChartTools._plot_fig(data, title, is_image, is_online)

    @staticmethod
    def plot_heat_map(data, title="", is_image=False, is_online=False):
        """
        plot scatter
        :param data: (x,y,z), example : (['a','b','c'],['a','b','c'],[[1, 20, 30 ],[20, 1, 60], [30, 60, 1]])
        :param title: title of current chat
        :param is_image: show html with download tip when true
        :param is_online:
        :return:
        """

        data = [Heatmap(x=data[0], y=data[1], z=data[2])]

        ChartTools._plot_fig(data, title, is_image, is_online)


class SKLTools:
    """
    SKlearn工具(API:http://scikit-learn.org/stable/modules/classes.html)
    包括: 工具类（DataFrameSelector,CategoricalEncoder,ChiMerge,WeightBagging,StackingBagging）
         方法（select_features_from_model,binary_classification_model_estimation,multi_classification_model_estimation,regression_model_estimation）
         评估方法（mean_squared_log_error,root_mean_squared_log_error）

    """

    @staticmethod
    def mean_squared_log_error(y_true, y_predict):
        """
        平均平方对数误差
        :param y_true:
        :param y_predict:
        :return: error
        """
        return mean_squared_error(np.log(y_true.astype(float)), np.log(y_predict))

    @staticmethod
    def root_mean_squared_log_error(y_true, y_predict):
        """
        平均平方对数误差平方
        :param y_true:
        :param y_predict:
        :return: error
        """
        return np.sqrt(SKLTools.mean_squared_log_error(y_true, y_predict))

    @staticmethod
    def select_features_from_model(model, x_train, y_train):
        """
        根据模型获得选择后的特征
        :param model:
        :param x_train: 训练集特征
        :param y_train: 训练集label
        :return: 被选择特征的列名（list）
        """

        model.fit(x_train, y_train)

        feature_selector = SelectFromModel(model, prefit=True)

        selected_cols = list(
            dict(filter(lambda x: x[1] == True, zip(x_train.columns, feature_selector.get_support()))).keys())

        return selected_cols

    @staticmethod
    def binary_classification_model_estimation(model, x_feature, y_label, cv=3, learning_curve_scoring=None):
        """
        二分类模型评估
        :param model:分类模型
        :param x_feature:
        :param y_label:
        :param cv:交叉验证数
        :param learning_curve_scoring: string, callable or None
           默认: None
            1、string类型：学习曲线评估方式  "accuracy" ,
            2、函数句柄：sklearn评估函数 make_scorer(my_scorer(y_label,y_predict))
            3、函数句柄：返回error的score方法my_scorer(estimator, X, y_label)
            4、None 不绘制学习曲线
        :return:
        """
        y_probas_forest = cross_val_predict(
            model, x_feature, y_label, cv=cv, method="predict_proba")
        y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class

        fpr_forest, tpr_forest, thresholds_forest = roc_curve(
            y_label, y_scores_forest)
        ChartTools.plot_scatter([(fpr_forest, tpr_forest)], traces=["ROC"], title="ROC")
        print("auc : %f" % roc_auc_score(y_label, y_scores_forest))

        y_predict = cross_val_predict(model, x_feature, y_label, cv=cv)
        precisions, recalls, thresholds = precision_recall_curve(y_label, y_predict)
        ChartTools.plot_scatter([(recalls, precisions)], traces=["PRC"], title="PRC")
        print('accuracy : %f' % accuracy_score(y_label, y_predict))
        print('recall : %f' % recall_score(y_label, y_predict))

        if learning_curve_scoring:
            train_sizes, train_scores, test_scores = learning_curve(
                model, x_feature, y_label, cv=cv, scoring=learning_curve_scoring)
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            title = learning_curve_scoring if (isinstance(learning_curve_scoring, str)) else "scoring"

            ChartTools.plot_scatter([(train_sizes, train_scores_mean), (train_sizes, test_scores_mean)],
                                    traces=["train_%s" % title, "test_%s" % title],
                                    title="LearningCurve")

    @staticmethod
    def multi_classification_model_estimation(model, x_feature, y_label, cv=3, learning_curve_scoring='accuracy'):
        """
        多分类模型评估
        :param model:分类模型
        :param x_feature:训练数据
        :param y_label:训练目标
        :param cv:交叉验证数
        :param learning_curve_scoring: 学习曲线评估方式（accuracy）,
            sklearn评估函数（make_scorer(my_scorer(y_label,y_predict))）
            或返回error的score方法my_scorer(estimator, X, y_label)
        :return:
        """
        y_predict = cross_val_predict(model, x_feature, y_label, cv=cv)
        print('acc:%s' % accuracy_score(y_label, y_predict))
        train_sizes, train_scores, test_scores = learning_curve(
            model, x_feature, y_label, cv=cv, scoring=learning_curve_scoring)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        title = learning_curve_scoring if (isinstance(learning_curve_scoring, str)) else "scoring"

        ChartTools.plot_scatter([(train_sizes, train_scores_mean), (train_sizes,
                                                                    test_scores_mean)],
                                traces=["train_%s" % title, "test_%s" % title],
                                title="LearningCurve")

    @staticmethod
    def regression_model_estimation(model, x_feature, y_label, cv=3, learning_curve_scoring='neg_mean_squared_error'):
        """
        回归模型评估
        :param model:回归模型
        :param x_feature:训练数据
        :param cv:交叉验证数
        :param y_label:训练目标
        :param learning_curve_scoring:学习曲线评估方式（neg_mean_squared_error）,
            sklearn评估函数（make_scorer(my_scorer(y_label,y_predict))）
            或返回error的score方法my_scorer(estimator, X, y_label)
        :return:
        """
        y_predict = cross_val_predict(model, x_feature, y_label, cv=cv)
        mse = SKLTools.mean_squared_log_error(y_label, y_predict)
        print('MSE(Log):%s' % mse)
        print('RMSE(Log):%s' % np.sqrt(mse))
        print('R2:%s' % r2_score(y_label, y_predict))

        if learning_curve_scoring:
            train_sizes, train_scores, test_scores = learning_curve(
                model, x_feature, y_label, cv=cv, scoring=learning_curve_scoring)
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            title = learning_curve_scoring if (isinstance(learning_curve_scoring, str)) else "scoring"

            ChartTools.plot_scatter([(train_sizes, train_scores_mean), (train_sizes, test_scores_mean)],
                                    traces=["train_%s" % title, "test_%s" % title],
                                    title="LearningCurve")

    class DataFrameSelector(BaseEstimator, TransformerMixin):
        """
        列选择器
        """

        def __init__(self, attribute_names):
            self.attribute_names = attribute_names

        def fit(self):
            return self

        def transform(self, X):
            return X[self.attribute_names].values

    class ChiMerge(BaseEstimator, TransformerMixin):
        """
        卡方分箱
        """

        def __init__(self, data_frame, class_col, max_number_intervals, threshold=4.61,
                     min_expected_value=0.5):
            """
            停止分箱条件：最小卡方值大于阈值时、分箱数小于最大分箱数
            :param data_frame:
            :param class_col: 类列(label)
            :param max_number_intervals: 最大分箱数
            :param threshold: 卡方阈值
            :param min_expected_value: 卡方检验计算时的最小期望值
            """

            if type(data_frame) != pd.DataFrame:
                print('错误: 数据必须为 pandas.DataFrame 类型')
                return

            self.data = data_frame

            self.sorted_data = None

            self.frequency_matrix = None

            self.frequency_matrix_intervals = None

            self.max_number_intervals = max_number_intervals

            self.min_expected_value = min_expected_value

            self.threshold = threshold

            self.class_col = class_col

            self.attribute_col = None

            # first intervals: unique class values
            self.unique_class_values = np.unique(self.data[self.class_col])
            # number of classes
            self.nclasses = len(self.unique_class_values)
            # degress of freedom (look at table)
            degrees_freedom = self.nclasses - 1

            print('数据信息:')
            print('- 类列类数: {}'.format(self.nclasses))
            print('- 最大区间数: {}'.format(self.max_number_intervals))
            print('- 自由度: {} (deprecated)'.format(degrees_freedom))
            print('- 卡方阈值: {}'.format(self.threshold))

        def _prepare_data(self):
            """
            对需要分箱的列作预处理
            :return:
            """

            self.sorted_data = self.data[[self.attribute_col, self.class_col]].sort_values(self.attribute_col)

            # first intervals: unique attribute values
            self.frequency_matrix_intervals = np.unique(self.sorted_data[self.attribute_col])

            # 通过交叉视图获取attribute-class统计
            cross_tab = pd.crosstab(self.sorted_data[self.attribute_col],
                                    self.sorted_data[self.class_col]).reset_index()
            self.frequency_matrix = np.array(cross_tab[self.unique_class_values])

            print('')
            print('初始化 列{} 区间:'.format(self.attribute_col))
            print('- 数据实际区间数: {}'.format(len(self.frequency_matrix_intervals)))
            # print('- 区间分隔点: {}'.format(self.frequency_matrix_intervals))

        def fit(self, attribute_col):
            """
            对需要分箱的列进行分箱
            :param attribute_col:
            :return:
            """

            self.attribute_col = attribute_col

            self._prepare_data()

            chitest = {}
            counter = 0
            smallest = -1

            while self.frequency_matrix.shape[0] > self.max_number_intervals:
                chitest = {}
                shape = self.frequency_matrix.shape
                for r in range(shape[0] - 1):
                    interval = r, r + 1
                    chi2 = AnalyzeTools.chi2_test(self.frequency_matrix[[interval], :][0], self.min_expected_value)
                    if chi2 not in chitest:
                        chitest[chi2] = []
                    chitest[chi2].append((interval))
                smallest = min(chitest.keys())
                biggest = max(chitest.keys())

                # 总结
                counter += 1
                # print('')
                # print(
                #     '第 {} 次迭代: 区间数:{} . 最小卡方值:{}, 最大卡方值:{}'.format(counter, self.frequency_matrix.shape[0],
                #                                                    smallest, biggest))
                # print('两两分区卡方值: {}'.format(chitest.keys()))

                # 合并操作
                if smallest <= self.threshold:
                    # print('合并分区(卡方值->分隔点降排索引): chi {} -> {}'.format(smallest, chitest[smallest]))
                    for (lower, upper) in list(
                            reversed(chitest[smallest])):
                        # 合并分区后一个合并至前一个
                        for col in range(
                                shape[1]):
                            self.frequency_matrix[lower, col] += self.frequency_matrix[
                                upper, col]
                        # 删除已合并的分区(分隔点的class统计)
                        self.frequency_matrix = np.delete(self.frequency_matrix, upper,
                                                          axis=0)
                        # 删除已合并的分区(分隔点)
                        self.frequency_matrix_intervals = np.delete(self.frequency_matrix_intervals, upper,
                                                                    axis=0)
                    # print('新分区分隔点: ({}):{}'.format(len(self.frequency_matrix_intervals),
                    #                                self.frequency_matrix_intervals))
                else:
                    break

            chitestvalues = chitest
            print('结束: 迭代{}次 ，(最小卡方值 {} 大于阈值 {})\n'.format(counter, smallest, self.threshold))

            print('总结：')
            print('{}{}'.format('分隔点: ', self.frequency_matrix_intervals))
            print('{}{}'.format('卡方值: ', ', '.join([
                '[{}-{}):{:5.1f}'.format(
                    self.frequency_matrix_intervals[v[0][0]],
                    '最大值' if v[0][1] > len(self.frequency_matrix_intervals) - 1 else self.frequency_matrix_intervals[
                        v[0][1]],
                    k)
                for k, v in sort_dict_by_value(chitestvalues, False)])))

            final_frequency = pd.DataFrame(self.frequency_matrix)

            final_frequency['bin'] = self.frequency_matrix_intervals

            print('{}\n{}'.format('区间分类频次统计：', final_frequency))

        def transform(self, data_set):
            """
            分箱转化
            :return: pandas.series
            """

            final_intervals = list(self.frequency_matrix_intervals)

            def get_cate(x):
                t = None
                for i in final_intervals:
                    if x < i:
                        t = final_intervals.index(i)
                        break
                    else:
                        t = final_intervals.index(i) + 1
                return t

            return data_set[self.attribute_col].map(lambda x: get_cate(x))

    class CategoricalEncoder(BaseEstimator, TransformerMixin):
        """Encode categorical features as a numeric array.
        The input to this transformer should be a matrix of integers or strings,
        denoting the values taken on by categorical (discrete) features.
        The features can be encoded using a one-hot aka one-of-K scheme
        (``encoding='onehot'``, the default) or converted to ordinal integers
        (``encoding='ordinal'``).
        This encoding is needed for feeding categorical data to many scikit-learn
        estimators, notably linear models and SVMs with the standard kernels.
        Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
        Parameters
        ----------
        encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
            The type of encoding to use (default is 'onehot'):
            - 'onehot': encode the features using a one-hot aka one-of-K scheme
              (or also called 'dummy' encoding). This creates a binary column for
              each category and returns a sparse matrix.
            - 'onehot-dense': the same as 'onehot' but returns a dense array
              instead of a sparse matrix.
            - 'ordinal': encode the features as ordinal integers. This results in
              a single column of integers (0 to n_categories - 1) per feature.
        categories : 'auto' or a list of lists/arrays of values.
            Categories (unique values) per feature:
            - 'auto' : Determine categories automatically from the training data.
            - list : ``categories[i]`` holds the categories expected in the ith
              column. The passed categories are sorted before encoding the data
              (used categories can be found in the ``categories_`` attribute).
        dtype : number type, default np.float64
            Desired dtype of output.
        handle_unknown : 'error' (default) or 'ignore'
            Whether to raise an error or ignore if a unknown categorical feature is
            present during transform (default is to raise). When this is parameter
            is set to 'ignore' and an unknown category is encountered during
            transform, the resulting one-hot encoded columns for this feature
            will be all zeros.
            Ignoring unknown categories is not supported for
            ``encoding='ordinal'``.

        Attributes
        ----------
        categories_ : list of arrays
            The categories of each feature determined during fitting. When
            categories were specified manually, this holds the sorted categories
            (in order corresponding with output of `transform`).

        Examples
        --------
        Given a dataset with three features and two samples, we let the encoder
        find the maximum value per feature and transform the data to a binary
        one-hot encoding.
        # >>> from sklearn.preprocessing import CategoricalEncoder
        # >>> enc = CategoricalEncoder(handle_unknown='ignore')
        # >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        # ... # doctest: +ELLIPSIS
        # CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
        #           encoding='onehot', handle_unknown='ignore')
        # >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
        array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
               [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])

        See also
        --------
        sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
          integer ordinal features. The ``OneHotEncoder assumes`` that input
          features take on values in the range ``[0, max(feature)]`` instead of
          using the unique values.
        sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
          dictionary items (also handles string-valued features).
        sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
          encoding of dictionary items or strings.
        """

        def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                     handle_unknown='error'):
            self.encoding = encoding
            self.categories = categories
            self.dtype = dtype
            self.handle_unknown = handle_unknown

        def fit(self, X):
            """Fit the CategoricalEncoder to X.
            Parameters
            ----------
            X : array-like, shape [n_samples, n_feature]
                The data to determine the categories of each feature.
            Returns
            -------
            self
            """

            if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
                template = ("encoding should be either 'onehot', 'onehot-dense' "
                            "or 'ordinal', got %s")
                raise ValueError(template % self.handle_unknown)

            if self.handle_unknown not in ['error', 'ignore']:
                template = ("handle_unknown should be either 'error' or "
                            "'ignore', got %s")
                raise ValueError(template % self.handle_unknown)

            if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
                raise ValueError("handle_unknown='ignore' is not supported for"

                                 " encoding='ordinal'")

            X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)

            n_samples, n_features = X.shape

            self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

            for i in range(n_features):
                le = self._label_encoders_[i]
                Xi = X[:, i]
                if self.categories == 'auto':
                    le.fit(Xi)
                else:
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        if self.handle_unknown == 'error':
                            diff = np.unique(Xi[~valid_mask])
                            msg = ("Found unknown categories {0} in column {1}"
                                   " during fit".format(diff, i))
                            raise ValueError(msg)
                    le.classes_ = np.array(np.sort(self.categories[i]))
            self.categories_ = [le.classes_ for le in self._label_encoders_]
            return self

        def transform(self, X):

            """Transform X using one-hot encoding.
            Parameters
            ----------
            X : array-like, shape [n_samples, n_features]
                The data to encode.
            Returns
            -------
            X_out : sparse matrix or a 2-d array
                Transformed input.
            """

            X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
            n_samples, n_features = X.shape
            X_int = np.zeros_like(X, dtype=np.int)
            X_mask = np.ones_like(X, dtype=np.bool)

            for i in range(n_features):
                valid_mask = np.in1d(X[:, i], self.categories_[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(X[~valid_mask, i])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during transform".format(diff, i))
                        raise ValueError(msg)
                    else:
                        # Set the problematic rows to an acceptable value and
                        # continue `The rows are marked `X_mask` and will be
                        # removed later.
                        X_mask[:, i] = valid_mask
                        X[:, i][~valid_mask] = self.categories_[i][0]
                X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

            if self.encoding == 'ordinal':
                return X_int.astype(self.dtype, copy=False)

            mask = X_mask.ravel()

            n_values = [cats.shape[0] for cats in self.categories_]
            n_values = np.array([0] + n_values)

            indices = np.cumsum(n_values)

            column_indices = (X_int + indices[:-1]).ravel()[mask]
            row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                    n_features)[mask]

            data = np.ones(n_samples * n_features)[mask]
            out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                    shape=(n_samples, indices[-1]),
                                    dtype=self.dtype).tocsr()

            if self.encoding == 'onehot-dense':
                return out.toarray()
            else:
                return out

    class WeightBagging(BaseEstimator, RegressorMixin):
        """
        权重集成
        :example
            w1 = 0.02
            w2 = 0.2
            w3 = 0.25
            w4 = 0.3
            w5 = 0.03
            w6 = 0.2
            weight_model = WeightBagging(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
        """

        def __init__(self, mod, weight):
            """
            初始化
            :param mod:模型实例列表
            :param weight: 模型权重列表（和为1）
            """
            self.mod = mod
            self.weight = weight
            self.models_ = None

        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.mod]
            for model in self.models_:
                model.fit(X, y)
            return self

        def predict(self, X):
            w = list()
            pred = np.array([model.predict(X) for model in self.models_])
            # for every data point, single model prediction times weight, then add them together
            for data in range(pred.shape[1]):
                single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
                w.append(np.sum(single))
            return w

    class StackingBagging(BaseEstimator, TransformerMixin):
        """
        stacking集成
        :example
            stack_model = StackingBagging(mod=[lgb,ela,svr],meta_model=lgb)
        """

        def __init__(self, mod, meta_model):
            """
            初始化
            :param mod: 模型实例列表
            :param meta_model: 基础（根）模型实例
            """
            self.mod = mod
            self.meta_model = meta_model
            self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
            self.saved_model = None

        def fit(self, X, y):

            X = Imputer().fit_transform(X)
            y = Imputer().fit_transform(y.values.reshape(-1, 1)).ravel()

            self.saved_model = [list() for i in self.mod]
            oof_train = np.zeros((X.shape[0], len(self.mod)))

            for i, model in enumerate(self.mod):
                for train_index, val_index in self.kf.split(X, y):
                    renew_model = clone(model)
                    renew_model.fit(X[train_index], y[train_index])
                    self.saved_model[i].append(renew_model)
                    oof_train[val_index, i] = renew_model.predict(X[val_index])

            self.meta_model.fit(oof_train, y)
            return self

        def predict(self, X):

            X = Imputer().fit_transform(X)
            whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                          for single_model in self.saved_model])
            return self.meta_model.predict(whole_test)

        def get_oof(self, X, y, test_X):
            """
            根据训练集验证集获得基础（根）模型训练数据
            :param X:
            :param y:
            :param test_X:
            :return:
            """

            oof = np.zeros((X.shape[0], len(self.mod)))
            test_single = np.zeros((test_X.shape[0], 5))
            test_mean = np.zeros((test_X.shape[0], len(self.mod)))
            for i, model in enumerate(self.mod):
                for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                    clone_model = clone(model)
                    clone_model.fit(X[train_index], y[train_index])
                    oof[val_index, i] = clone_model.predict(X[val_index])
                    test_single[:, j] = clone_model.predict(test_X)
                test_mean[:, i] = test_single.mean(axis=1)
            return oof, test_mean


class AnalyzeTools:
    """
    数据分析工具
    包括: freq_analyze,chi_merge,show_null_feature_rank,dummies;z_scores_std,show_confusion_matrix
    """

    @staticmethod
    def freq_analyze(data_frame, field_col='', target_col='', barmode='stack'):
        """
        频率分布分析
        :param data_frame:数据源
        :param field_col:需要分析的字段
        :param target_col:分类字段
        :param barmode:统计条显示方式
        :return:
        """
        if field_col is not '':
            data = []
            if target_col is not '':
                class_tag = list(data_frame[target_col].drop_duplicates())
                for cls in class_tag:
                    freq_data_part = eval('data_frame[data_frame.%s == %s]' % (target_col, cls))
                    freq_data = pd.DataFrame(freq_data_part[field_col].value_counts()).reset_index()
                    data.append((freq_data['index'], freq_data[field_col]))

                ChartTools.plot_bar(data=data, traces=class_tag, title="Frequency_Analyze", barmode=barmode)
            else:
                freq_data = pd.DataFrame(data_frame[field_col].value_counts()).reset_index()
                data.append((freq_data['index'], freq_data[field_col]))
                ChartTools.plot_bar(data=data, traces=[field_col], title="Frequency_Analyze", barmode=barmode)

    @staticmethod
    def chi2_test(array, min_expected_value):
        """
        对观测结果做卡方校验
        :param min_expected_value:期望值
        :param array: np.array 2 consecutive rows from frequeny attribute/class matrix, e.g.,: a = np.matrix('16 0 0; 4 1 1')
        :return chisqr value of distribution of 2 rows
        """

        shape = array.shape
        N = float(array.sum())  # total number of observations

        r = {}
        for i in range(shape[0]):
            r[i] = array[i].sum()

        c = {}
        for j in range(shape[1]):
            c[j] = array[:, j].sum()

        chi2 = 0
        for row in range(shape[0]):
            for col in range(shape[1]):
                e = r[row] * c[col] / N  # expected value
                o = array[row, col]  # observed value
                e = min_expected_value if e < min_expected_value else e
                chi2 += 0. if e == 0. else np.math.pow((o - e), 2) / float(e)

        return chi2

    @staticmethod
    def show_feature_importance(feature_list, importance_list, return_importance_df=False):
        """
        统计特征重要性
        :param feature_list:特征名列表
        :param importance_list:特征重要性列表（与特征名列表同序）
        :param return_importance_df:是否返回重要性表
        :return: feature_df
        """

        feature_df = pd.DataFrame()
        feature_df['feature'] = feature_list
        feature_df['importance'] = importance_list
        feature_df.sort_values(by=['importance'], ascending=True, inplace=True)

        ChartTools.plot_bar([(feature_df['importance'], feature_df['feature'])], traces=['importance'],
                            title='feature_importance', orientation='h')

        if return_importance_df:
            return feature_df

    @staticmethod
    def show_confusion_matrix(y_predict, y_true, col_name, normalize=False):
        """
        绘制并返回混淆矩阵
        :param y_predict:
        :param y_true:
        :param col_name:分类列名
        :param normalize:是否归一化
        :return: confusion matrix
        """
        cnf_matrix = confusion_matrix(y_true, y_predict)
        np.set_printoptions(precision=2)

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"

        x = list(map(lambda i: "predict_%s" % i, col_name))
        y = list(map(lambda i: "true_%s" % i, col_name))
        ChartTools.plot_heat_map((x, y, cnf_matrix), title)
        return cnf_matrix

    @staticmethod
    def show_null_feature_rank(data_frame):
        """
        显示包含空值特征排行
        :param data_frame:pandas df
        :return:pandas series
        """
        temp = data_frame.isnull().sum()
        return temp[temp > 0].sort_values(ascending=False)

    @staticmethod
    def dummies(data_frame, cols, drop_cols=True):
        """
        对指定列进行哑编码
        :param data_frame:
        :param cols:
        :param drop_cols: 是否删除原列
        :return: 哑编码后的数据集
        """
        for col in cols:
            dum = pd.get_dummies(data_frame[col], prefix=col)
            data_frame = pd.concat([data_frame, dum], axis=1)
            if drop_cols:
                data_frame.drop(col, axis=1, inplace=True)
        return data_frame

    @staticmethod
    def z_scores_std(data_frame, cols):
        """
        Z-Scores标准化
        :param data_frame:
        :param cols:
        :return: 标准化的数据集
        """
        for col in cols:
            data_frame[col] = data_frame[col].astype(float)
            data_frame[col] -= data_frame[col].mean()
            data_frame[col] /= data_frame[col].std()
        return data_frame

    @staticmethod
    def similarity(a, b, n=4):
        """
        基于n-grams的jaccard相似度
        :param a:
        :param b:
        :param n:
        :return:
        """

        a = set([a[i:i + n] for i in range(len(a) - n)])
        b = set([b[i:i + n] for i in range(len(b) - n)])
        a_and_b = a & b
        if not a_and_b:
            return 0.
        a_or_b = a | b
        return 1. * len(a_and_b) / len(a_or_b)

if __name__ == '__main__':
    # Scatter
    # t = np.array([0, 0.7, 1])
    # s = np.sin(2 * np.pi * t)
    # t1 = np.array([0, 0.25, 0.5, 0.88, 0.9, 1])
    # s1 = np.sin(2 * np.pi * t1)
    # t2 = np.array([0, 0.3, 0.6, 0.7, 1])
    # s2 = np.sin(2 * np.pi * t2)
    # ChartTools.plot_scatter([(t, s), (t1, s1), (t2, s2)], ["sin", "sin1", "sin2"], title='test', is_online=False)

    # Bar
    # ChartTools.plot_bar([(['a', 'b', 'c'], [10, 20, 30]), (['a', 'b', 'c'], [14, 15, 18])], ["food", "water"],
    # title='test', is_online=False)

    # Box
    # y0 = np.random.randn(50) - 1
    # y1 = np.random.randn(50) + 1
    # ChartTools.plot_box([y0,y1],['trace0','trace1'])

    # Pie
    # ChartTools.plot_pie((['a', 'b', 'c'], [14, 15, 18]),title='test', is_online=False)

    # Scatter3d
    # x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 200).transpose()
    # x2, y2, z2 = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 200).transpose()
    # ChartTools.plot_scatter3d([(x, y, z), (x2, y2, z2)], ["sample1", "sample2"], title='test')

    # Image to np.array
    # im = ImageTools.read_image("/home/raven/slice0.png")
    # arr = ImageTools.to_np_array(im)
    # print(arr.shape)
    # print(arr[0, :, :])
    # im2 = ImageTools.to_pil_image(arr)
    # ImageTools.save_image(im,"/home/raven/slice2.png")

    # Image RBG histogram
    # im = ImageTools.read_image("/home/raven/slice0.png")
    # im_gray = ImageTools.to_gray_image(im)
    # im_list = im.histogram()
    # im_gray_list = im_gray.histogram()
    # ChartTools.plot_area([(list(range(0, len(im_list))), im_list), (list(range(0, len(im_gray_list))), im_gray_list)],
    #                      ["im", "im_gray_list"], title='im_histogram', is_online=False)

    # Image slice
    # im = ImageTools.read_image("/home/raven/slice0.png")
    # slicer = ImageTools.ImageSlicer(im, (50, 30))
    # num = 0
    # for image,pos in slicer:
    #     num += 1
    #     ImageTools.save_image(image, "/home/raven/slice_{0}_{1}_{2}.png".format(pos[0],pos[1], num))
    # print(len(slicer))

    # Image combiner
    # im = ImageTools.read_image("/home/raven/slice0.png")
    # ic = ImageTools.ImageCombiner(im.size)
    # width = 50
    # height = 50
    # slicer = ImageTools.ImageSlicer(im, (width, height))
    # for image, pos in slicer:
    #     ic.append(image, pos)
    # ImageTools.save_image(ic.finish(), "/home/raven/slice1.png")

    list_file("data")
