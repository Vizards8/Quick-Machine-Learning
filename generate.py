import pandas as pd
import numpy as np
from sklearn.metrics import *
from codes.evaluation import *
from sklearn.model_selection import train_test_split
import os,time

from sklearn.linear_model import LinearRegression
def plot_ROC_curve(model_name, y_test, y_predict):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    if os.path.isfile('./static/modelresult/' + model_name + '_ROC.png'):
        os.remove('./static/modelresult/' + model_name + '_ROC.png')
    plt.savefig('./static/modelresult/' + model_name + '_ROC.png')  # 此处可以回传
    plt.close()
    # plt.show()

def plot_confusion_matrix(model_name,cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if os.path.isfile('./static/modelresult/' + model_name + '_confusion.png'):
        os.remove('./static/modelresult/'+model_name+'_confusion.png')
    plt.savefig('./static/modelresult/'+model_name+'_confusion.png') #此处可以回传
    plt.close()
    # plt.show()

FILE_PATH=['flex.csv', 'punch.csv']

FILE_PATH = np.unique(FILE_PATH)

FEATURES=['features']

TARGET='target'

MODEL=LinearRegression()

model_name=str(MODEL)[0:-2]

        # FILE_PATH={}#dataset_name
# FEATURES={}
# TARGET={}
# MODEL={}
# if FILE_PATH.split('.')[-1]=='xls' or FILE_PATH.split('.')[-1]=='xlsx':
#     DF=pd.read_excel(FILE_PATH)
# else:
#     DF=pd.read_csv(FILE_PATH)
#
# y =DF[TARGET]
# X = DF[FEATURES]

# 读取数据集
data = []
label = []


def readdata(_label, _filename):
    temp = []
    filedata = pd.read_csv(path + _filename, keep_default_na=False)
    filedata = np.array(filedata)
    for i in filedata:
        if i.all() != '':
            temp.append(i)
        else:
            data.append(temp)
            temp = []
            label.append(_label)


pre_time = time.time()
path = './Datasets/'
for _label, i in enumerate(FILE_PATH):
    readdata(_label, i)

data = np.array(data)
label = np.array(label)
# print(label.shape)

# 划分数据集
data = data.reshape(data.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

# 模型训练
model = MODEL.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 模型评估
# 绘制混淆矩阵
y_pred = [int(i > 0.5) for i in y_pred]
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=len(np.unique(label)))  # 设置打印数量的阈值
class_names = np.unique(label)
test_report = classification_report(y_test, y_pred)
# print(test_report)

timeused = time.time() - pre_time
acc = accuracy_score(y_test, y_pred)
with open('./static/modelresult/train_result.txt', 'a+') as f:
    f.write(model_name + '*' + str(timeused)[0:10] + 's*' + str(acc*100) + '%*' + './static/modelresult/' + model_name + '_confusion.png' + '*')

plot_confusion_matrix(model_name, cnf_matrix, classes=class_names, title='Confusion matrix')
plot_ROC_curve(model_name, y_test, y_pred)
f = open('./static/modelresult/' + model_name + 'test_report.txt', 'w')
f.write(test_report)
f.close()

