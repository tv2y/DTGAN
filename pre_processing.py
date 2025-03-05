import random
import numpy
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing, manifold, metrics, svm, tree
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils import data as Data
from torch.utils.data import DataLoader
from Network_Model import Classifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 读取训练集、测试集数据
train_raw_data = pd.read_csv(r"D:\tv2y\code\experiment2\Data\NSL_KDD_Dataset\trainset.csv", header=None, low_memory=False)
test_raw_data = pd.read_csv(r"D:\tv2y\code\experiment2\Data\NSL_KDD_Dataset\testset.csv", header=None, low_memory=False)
# train_raw_data = pd.read_csv(r"/home/baiwuxia/project/Adversarial Example/GAN Based Protetion/Data/NSL_KDD_Dataset/trainset.csv", header=None, low_memory=False)
# test_raw_data = pd.read_csv(r"/home/baiwuxia/project/Adversarial Example/GAN Based Protetion/Data/NSL_KDD_Dataset/testset.csv", header=None, low_memory=False)

feature_num = 41

# 读取前41列数据赋值给train_x，读取最后一列数据赋值给train_y
train_x = train_raw_data.iloc[:, :feature_num]
train_y = train_raw_data.iloc[:, feature_num]

test_x = test_raw_data.iloc[:, :feature_num]
test_y = test_raw_data.iloc[:, feature_num]

five_attack_type = ['benign', 'dos', 'probe', 'r2l', 'u2r']
two_attack_type = ['Normal', 'Attack']

data_encoder = preprocessing.OrdinalEncoder()
label_encoder = preprocessing.OneHotEncoder()
min_max_scaler = preprocessing.MinMaxScaler()

# 训练集数据编码、归一化处理
train_x = data_encoder.fit_transform(train_x)
train_x = min_max_scaler.fit_transform(train_x)
# 将五种类型划分成攻击和非攻击类型，攻击类型标记为1，非攻击类型标记为0
for i in range(len(five_attack_type)):
    if i == 0:
        train_y[train_y == five_attack_type[i]] = 0  # one-hot编码后对应(1,0)
    else:
        train_y[train_y == five_attack_type[i]] = 1  # one-hot编码后对应(0,1)
train_y = label_encoder.fit_transform(train_y.values.reshape(-1, 1)).todense()

# 测试集数据编码、归一化处理
test_x = data_encoder.fit_transform(test_x)
test_x = min_max_scaler.fit_transform(test_x)
for i in range(len(five_attack_type)):
    if i == 0:
        test_y[test_y == five_attack_type[i]] = 0  # one-hot编码后对应(1,0)
    else:
        test_y[test_y == five_attack_type[i]] = 1  # one-hot编码后对应(0,1)
test_y = label_encoder.fit_transform(test_y.values.reshape(-1, 1)).todense()

batch_size = 1024
seed = 1024

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

setup_seed(seed)

'''
将训练集的所有数据打包成一个dataloader
'''
def Load_Trainset_To_1_Dataloader():
    global train_x, train_y
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_dataset.data = train_dataset.tensors[0]
    train_dataset.target = train_dataset.tensors[1]
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    print("total trainset dataloader: ", len(train_x))
    return train_dataloader

'''
将训练集的所有数据打包成两个dataloader
'''
def Load_Trainset_To_2_Dataloaders():
    global train_x, train_y
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)

    # 将训练集数据分成两个部分
    split_index = len(train_x) // 2
    train_x1, train_x2 = train_x[:split_index], train_x[split_index:]
    train_y1, train_y2 = train_y[:split_index], train_y[split_index:]

    # 打包第一个数据集
    train_dataset1 = Data.TensorDataset(train_x1, train_y1)
    train_dataloader1 = DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=True)
    print("Total samples in train dataset 1:", len(train_x1))

    # 打包第二个数据集
    train_dataset2 = Data.TensorDataset(train_x2, train_y2)
    train_dataloader2 = DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=True)
    print("Total samples in train dataset 2:", len(train_x2))

    return train_dataloader1, train_dataloader2

'''
将测试集所有数据打包成一个dataloader
'''
def Load_Testset_To_1_Dataloader():
    global test_x, test_y
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    test_dataset.data = test_dataset.tensors[0]
    test_dataset.target = test_dataset.tensors[1]
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print("total testset dataloader: ", len(test_x))
    return test_dataloader

'''
将测试集所有数据打包成两个dataloader
'''
def Load_Testset_To_2_Dataloaders():
    global test_x, test_y
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)

    # 将测试集数据分成两个部分
    split_index = len(test_x) // 2
    test_x1, test_x2 = test_x[:split_index], test_x[split_index:]
    test_y1, test_y2 = test_y[:split_index], test_y[split_index:]

    # 打包第一个数据集
    test_dataset1 = Data.TensorDataset(test_x1, test_y1)
    test_dataloader1 = DataLoader(dataset=test_dataset1, batch_size=batch_size, shuffle=False)
    print("Total samples in test dataset 1:", len(test_x1))

    # 打包第二个数据集
    test_dataset2 = Data.TensorDataset(test_x2, test_y2)
    test_dataloader2 = DataLoader(dataset=test_dataset2, batch_size=batch_size, shuffle=False)
    print("Total samples in test dataset 2:", len(test_x2))

    return test_dataloader1, test_dataloader2

'''
将两个dataloader合并成一个
'''
def Merge_Two_Dataloader_To_One(dataloader1, dataloader2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data1 = dataloader1.dataset.data.to(device)
    data2 = dataloader2.dataset.data.to(device)
    target1 = dataloader1.dataset.target.to(device)
    target2 = dataloader2.dataset.target.to(device)
    data = torch.cat((data1, data2), 0)	# 在 0 维(纵向)进行拼接
    target = torch.cat((target1, target2), 0)
    dataset = Data.TensorDataset(data, target)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=False)
    return dataloader


'''
选择训练集中攻击/非攻击类型的一种数据打包成dataloader
'''
def Load_One_Type_Trainset(type):
    type_num = 1 if type == "attack" else 0
    global train_x, train_y
    # 根据train_y矩阵获取攻击类型数据的列索引
    # 当最大值位置在第一个元素时，即(1,0)，其对应的target为0，即非攻击类型；当最大值位置在第二个元素时，即(0,1)，其对应的target为1，即攻击类型
    attack_indexes_for_y = np.argmax(train_y, axis=1)  # 按行索引train_y中每一行的最大值
    # 根据攻击类型的列索引获取相应的行索引，从而获取对应的x
    attack_indexes_for_x = np.where(attack_indexes_for_y==type_num)  # 返回的是一个元组
    data = train_x[attack_indexes_for_x[0]]
    label = train_y[attack_indexes_for_x[0]]
    data = torch.FloatTensor(data)
    label = torch.FloatTensor(label)
    # 打包成dataloader
    dataset = Data.TensorDataset(data, label)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader

'''
选择训练集中攻击/非攻击类型的一种数据打包成两个dataloader
'''
def Load_One_Type_To_Two_Trainset(type):
    type_num = 1 if type == "attack" else 0
    global train_x, train_y
    # 根据train_y矩阵获取攻击类型数据的列索引
    # 当最大值位置在第一个元素时，即(1,0)，其对应的target为0，即非攻击类型；当最大值位置在第二个元素时，即(0,1)，其对应的target为1，即攻击类型
    attack_indexes_for_y = np.argmax(train_y, axis=1)  # 按行索引train_y中每一行的最大值
    # 根据攻击类型的列索引获取相应的行索引，从而获取对应的x
    attack_indexes_for_x = np.where(attack_indexes_for_y == type_num)  # 返回的是一个元组
    data = train_x[attack_indexes_for_x[0]]
    label = train_y[attack_indexes_for_x[0]]
    data = torch.FloatTensor(data)
    label = torch.FloatTensor(label)
    # 创建两个新的dataset对象
    dataset1 = Data.TensorDataset(data[:len(data)//2], label[:len(data)//2])
    dataset2 = Data.TensorDataset(data[len(data)//2:], label[len(data)//2:])
    # 设置dataset1的data和target属性
    dataset1.data = dataset1.tensors[0]
    dataset1.target = dataset1.tensors[1]
    # 设置dataset2的data和target属性
    dataset2.data = dataset2.tensors[0]
    dataset2.target = dataset2.tensors[1]
    # 创建两个新的dataloader对象
    dataloader1 = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=False)
    dataloader2 = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=False)
    return dataloader1, dataloader2

'''
选择训练集中攻击/非攻击类型的一种数据打包成dataloader
'''
def Load_One_Type_Testset(binary_type):
    type_num = 1 if binary_type == "attack" else 0
    global test_x, test_y
    # 根据train_y矩阵获取攻击类型数据的列索引
    # 当最大值位置在第一个元素时，即(1,0)，其对应的target为0，即非攻击类型；当最大值位置在第二个元素时，即(0,1)，其对应的target为1，即攻击类型
    attack_indexes_for_y = np.argmax(test_y, axis=1)  # 按行索引train_y中每一行的最大值
    # 根据攻击类型的列索引获取相应的行索引，从而获取对应的x
    attack_indexes_for_x = np.where(attack_indexes_for_y==type_num)  # 返回的是一个元组
    data = test_x[attack_indexes_for_x[0]]
    label = test_y[attack_indexes_for_x[0]]
    data = torch.FloatTensor(data)
    label = torch.FloatTensor(label)
    # 打包成dataloader
    dataset = Data.TensorDataset(data, label)
    dataset.data = dataset.tensors[0]
    dataset.target = dataset.tensors[1]
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def train_classifier_on_trainset(train_dataloader, epoch):
    test_dataloader = Load_Testset_To_1_Dataloader()
    classifier = Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
    global_step = 0
    for i in range(epoch):
        train_accuracy_sum = 0
        train_loss_sum = 0
        count = len(train_dataloader.dataset)
        for data, targets in train_dataloader:
            data = data.to(device)

            targets = targets.to(device)

            optimizer.zero_grad()

            output = classifier(data).to(device)

            loss = loss_fn(output, targets).sum()

            train_loss_sum += loss.item()

            loss.backward()

            optimizer.step()

            train_accuracy_sum += (output.argmax(1)==targets.argmax(1)).sum().item()

        train_accuracy = train_accuracy_sum/count
        print("epoch:", i, ", train_loss: ", "%.6f" % (train_loss_sum/count), ", train accuracy: ", "%.6f" % train_accuracy)
        # writer.add_scalar("train_loss: ", train_loss_sum/count, global_step)
        # writer.add_scalar("train_accuracy: ", train_accuracy, global_step)

        # test_classifier_on_test(test_dataloader, classifier)

        global_step += 1

    # torch.save(classifier, "model/classification_model_"+"_"+str(start)+".pth")

    output = classifier(train_dataloader.dataset.data.to(device)).to('cpu')
    cf_matrix = metrics.confusion_matrix(train_dataloader.dataset.target.argmax(1).to('cpu'),
                                         output.argmax(1).detach().to('cpu'))
    print(cf_matrix)

    return classifier


def test_classifier_on_test(test_dataloader, classifier):
    torch.set_printoptions(threshold=numpy.inf, linewidth=400)
    numpy.set_printoptions(threshold=numpy.inf, linewidth=400)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataloader.dataset.data = test_dataloader.dataset.data.to(device)
    test_dataloader.dataset.target = test_dataloader.dataset.target.to(device)
    count = len(test_dataloader.dataset)
    output = classifier(test_dataloader.dataset.data).to(device)
    cf_matrix = metrics.confusion_matrix(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'))
    precision_5 = metrics.precision_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'), average=None)
    average_precision_weighted = metrics.precision_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'), average='weighted')
    recall_5 = metrics.recall_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'), average=None)
    average_recall_weighted = metrics.recall_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'), average='weighted')
    f1_5 = metrics.f1_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'), average=None)
    average_f1_weighted = metrics.f1_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'), average='weighted')
    accuracy = metrics.accuracy_score(test_dataloader.dataset.target.argmax(1).to('cpu'),
                                 output.argmax(1).detach().to('cpu'))
    # roc_auc_5 = metrics.roc_auc_score(test_dataloader.dataset.target.to('cpu'),
    #                              output.detach().to('cpu'), average=None, multi_class='ovo')
    # average_roc_auc_weighted = metrics.roc_auc_score(test_dataloader.dataset.target.to('cpu'),
    #                              output.detach().to('cpu'), average='weighted', multi_class='ovo')
    print(cf_matrix)

    # metrics.plot_confusion_matrix(classifier, test_dataloader.dataset.Data, test_dataloader.dataset.target)
    for i in range(len(two_attack_type)):
        index = numpy.argwhere(test_dataloader.dataset.target.argmax(1).to('cpu') == i).flatten()
        print('%-25s'%two_attack_type[i], "\t", ":", "%4d" % len(index), "\tprecision:", "%.4f" % precision_5[i], "\trecall:",
              "%.4f" % recall_5[i], "\tf1:", "%.4f" % f1_5[i], ) # "\troc_auc:", "%.4f" % roc_auc_5[i]
    print("weighted\naverage", ":", "%.4f" % accuracy, "\tprecision:", "%.4f" % average_precision_weighted, "\trecall:",
          "%.4f" % average_recall_weighted, "\tf1:", "%.4f" % average_f1_weighted, ) # "\troc_auc:", "%.4f" % average_roc_auc_weighted, average_roc_auc_weighted
    return cf_matrix, average_precision_weighted, average_recall_weighted, average_f1_weighted

def print_result(y_test, output):
    y_test = torch.tensor(y_test)
    output = torch.tensor(output)
    cf_matrix = metrics.confusion_matrix(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'))
    precision_5 = metrics.precision_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'), average=None)
    average_precision_weighted = metrics.precision_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'), average='weighted')
    recall_5 = metrics.recall_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'), average=None)
    average_recall_weighted = metrics.recall_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'), average='weighted')
    f1_5 = metrics.f1_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'), average=None)
    average_f1_weighted = metrics.f1_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'), average='weighted')
    accuracy = metrics.accuracy_score(y_test.argmax(1).to('cpu'), output.argmax(1).to('cpu'))
    roc_auc_5 = metrics.roc_auc_score(y_test.to('cpu'), output.to('cpu'), average=None, multi_class='ovo')
    average_roc_auc_weighted = metrics.roc_auc_score(y_test.to('cpu'), output.to('cpu'), average='weighted', multi_class='ovo')
    print(cf_matrix)
    # metrics.plot_confusion_matrix(classifier, test_dataloader.dataset.Data, test_dataloader.dataset.target)
    for i in range(len(five_attack_type)):
        index = numpy.argwhere(y_test.argmax(1).to('cpu') == i).flatten()
        print(five_attack_type[i], "\t", ":", "%4d" % len(index), "\tprecision:", "%.4f" % precision_5[i], "\trecall:",
              "%.4f" % recall_5[i], "\tf1:", "%.4f" % f1_5[i], "\troc_auc:", "%.4f" % roc_auc_5[i])
    print("weighted\naverage", ":", "%.4f" % accuracy, "\tprecision:", "%.4f" % average_precision_weighted, "\trecall:",
          "%.4f" % average_recall_weighted, "\tf1:", "%.4f" % average_f1_weighted, "\troc_auc:",
          "%.4f" % average_roc_auc_weighted)
    return average_precision_weighted, average_recall_weighted, average_f1_weighted, average_roc_auc_weighted


def RF(train_dataloader, test_dataloader, seed):
    x_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.target
    x_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.target
    regressor = RandomForestRegressor(n_estimators=10, max_depth=12, min_samples_split=28, min_samples_leaf=13, random_state=seed)
    regressor.fit(x_train, y_train)
    output = regressor.predict(x_test)
    return print_result(y_test, output)


def DT(train_dataloader, test_dataloader, seed):
    x_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.target
    x_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.target
    dt = tree.DecisionTreeClassifier(splitter='best', criterion='gini', max_depth=9, min_samples_split=31,
                                     min_samples_leaf=4, min_weight_fraction_leaf=6.2e-05, min_impurity_decrease=4.1e-06,
                                     max_leaf_nodes=30, random_state=seed)
    dt.fit(x_train, y_train)
    output = dt.predict(x_test)
    return print_result(y_test, output)


def SVM(train_dataloader, test_dataloader, seed):
    x_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.target.argmax(1)
    x_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.target
    s = svm.SVC(C=99.83898, kernel='poly', random_state=seed)
    s.fit(x_train, y_train)
    output = s.predict(x_test)
    output = label_encoder.fit_transform(output.reshape(-1, 1)).todense()
    return print_result(y_test, output)


def KNN(train_dataloader, test_dataloader):
    x_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.target
    x_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.target
    knn = KNeighborsClassifier(algorithm='brute', n_neighbors=10, weights='uniform', p=2, leaf_size=2)
    knn.fit(x_train, y_train)
    output = knn.predict(x_test)
    return print_result(y_test, output)


# def AdaBoostClassifier
def AdaBoost(train_dataloader, test_dataloader):
    x_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.target
    x_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.target
    adaboost = AdaBoostClassifier()
    adaboost.fit(x_train, y_train)
    output = adaboost.predict(x_test)
    return print_result(y_test, output)


def XGB(train_dataloader, test_dataloader, seed):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 5,  # 类别数，与 multisoftmax 并用
        'gamma': 0.3336,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 3,  # 构建树的深度，越大越容易过拟合
        'lambda': 0.0022,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'alpha': 1.6e-06,
        'subsample': 0.3485,  # 随机采样训练样本
        'colsample_bytree': 0.9840,  # 生成树时进行的列采样
        'n_estimators':96,
        'min_child_weight': 3,
        'eta': 0.7437,  # 如同学习率
        'seed': seed,
        'grow_policy': 'lossguide',
    }
    x_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.target.argmax(1)
    x_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.target
    dtrain = xgboost.DMatrix(x_train, label=y_train)
    dtest = xgboost.DMatrix(x_test)
    bst = xgboost.train(params, dtrain)  # 训练
    # make prediction
    output = bst.predict(dtest)  # 预测
    output = label_encoder.fit_transform(output.reshape(-1, 1)).todense()
    # xgb = xgboost.XGBClassifier()
    # xgb.fit(x_train, y_train)
    # output = xgb.predict(x_test)
    return print_result(y_test, output)
