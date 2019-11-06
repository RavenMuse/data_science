# Data Science
[TOC]

## Deep Learning

### 1、Keras Example

**目录：** example/keras

### 2、Tensorflow Example

**目录：** example/tensorflow


## Kaggle competitions

### 1、Titanic: Machine Learning from Disaster

**目录：** kaggle/kaggle_titanic

**比赛说明：** 根据Titanic游客信息对预测游客遇难是否存活

**数据源：** kaggle competitions download -c titanic

**最后提交结果（accuracy）：**  0.80382

**特征工程相关：**
> * one-hot 编码
> * 训练模型填充空值
> * DNN迁移学习融合特征

**模型相关：**
> * Xgboost
> * Random Forest
> * K-Fold投票集成

### 2、House Prices: Advanced Regression Techniques

**目录：** kaggle/kaggle_house_prices

**比赛说明：** 根据住房相关特征预测住房价格

**数据源：** kaggle competitions download -c house-prices-advanced-regression-techniques

**最后提交结果（MSE）：**  0.14242

**特征工程相关：**
> * 填充空值
> * 组合特征
> * 数值、类型分箱
> * one-hot 编码
> * PCA 降维
> * 降噪（异常值处理）
> * RandomForestRegressor特征选择


**模型相关：**
> * Ridge、Lasso、GradientBoostingRegressor、XGBRegressor、SVR、LinearSVR、ElasticNet、BayesianRidge、ExtraTreesRegressor、LGBMRegressor、KernelRidge
> * 模型调优、模型选择
> * 多模型栈集成


### 3、Home Credit Default Risk

**分支：** kaggle/kaggle_home_credit_default_risk

**比赛说明：** 根据用户信贷数据判断用户是否具有信贷危机

**数据源：** kaggle competitions download -c home-credit-default-risk

**最后提交结果（ROC）：**  无

**特征工程相关：**
> * 表聚合、关联
> * 空值填充
> * 异常处理
> * 类型编码
> * 时间惩罚


**模型相关：**

