import os, json

'''
流程： 
1、从monodb读取用户上传的数据集
2、传入需要处理的列(特征列)，以及处理过程函数。 
   内置填充缺失值、删除缺失值以及one-hot编码
   标准化数据（MinMax），可选参数
3、模型构建，传入模型名称、目标列进行构建
   允许用户进行参数搜索，前端用户输入的搜索参数格式为json格式

主函数生成：
- 自动生成代码的代码模块存放于同一个文件夹：
   仅包含功能函数
- 主函数预先定义一个代码文件，相关参数通过占位符填充，填充的参数来源于前段输入，包括：主要包括特征列、目标列，文件名
'''

'''查看模型中文名用'''
'''
MODEL_DICT = {
    '分类': {
        '朴素贝叶斯': 'from sklearn.naive_bayes import GaussianNB',
        '决策树': 'from sklearn.tree import DecisionTreeClassifier',
        '支持向量机': 'from sklearn.svm import SVC',
        '神经网络': 'from sklearn.neural_network import MLPClassifier'
    },
    '回归': {
        '线性回归': 'from sklearn.linear_model import LinearRegression',
        '逻辑回归': 'from sklearn.linear_model import LogisticRegression',
        '决策树': 'from sklearn.tree import DecisionTreeRegressor',
        '支持向量机': 'from sklearn.svm import SVR',
        '神经网络': 'from sklearn.neural_network import MLPRegressor'
    },
    '聚类': {
        'K-means': 'from sklearn.cluster import KMeans'
    },
    'ROC曲线': 'plot_ROC_curve.py',
    '混淆矩阵': 'plot_confusion_matrix.py'
}
'''

MODEL_DICT_name = {
    '1': {
        '101': 'GaussianNB',
        '102': 'DecisionTreeClassifier',
        '103': 'SVM',
        '104': 'MLPClassifier'
    },
    '2': {
        '201': 'LinearRegression',
        '202': 'LogisticRegression',
        '203': 'DecisionTreeRegressor',
        '204': 'SVM',
        '205': 'MLPRegressor'
    },
    '3': {
        '301': 'K-Means'
    },
    '401': 'plot_ROC_curve.py',
    '402': 'plot_confusion_matrix.py'
}

MODEL_DICT = {
    '1': {
        '101': 'from sklearn.naive_bayes import GaussianNB',
        '102': 'from sklearn.tree import DecisionTreeClassifier',
        '103': 'from sklearn.svm import SVC',
        '104': 'from sklearn.neural_network import MLPClassifier'
    },
    '2': {
        '201': 'from sklearn.linear_model import LinearRegression',
        '202': 'from sklearn.linear_model import LogisticRegression',
        '203': 'from sklearn.tree import DecisionTreeRegressor',
        '204': 'from sklearn.svm import SVR',
        '205': 'from sklearn.neural_network import MLPRegressor'
    },
    '3': {
        '301': 'from sklearn.cluster import KMeans'
    },
    '401': 'plot_ROC_curve.py',
    '402': 'plot_confusion_matrix.py'
}


class SetModel():
    '''

    用于与前端界面交互，获取特征列，以及数据处理步骤。
    根据用户选择的步骤，读取预定义的代码。
    前端返回参数
    {
        target:[],
        features:[],

    }
    '''

    def __init__(self, dataset_name, target, features, model_type, model_name, evaluate_methods):
        '''
        参数与前端的用户输入一致
        '''
        self.code_files = './codes/'
        self.dataset_name = dataset_name
        self.target = target
        self.features = features.split(';')
        self.model_type = model_type
        self.model_name = model_name
        self.generate = ''
        self.evaluate_methods = evaluate_methods

    def clean_data(self, df, cols, op, standard=''):
        '''
           自动数据清洗
           df:
           cols:
           op:数据清洗的操作
           '''
        if op == 'fillna':
            df.loc[:, cols].fillna()
        elif op == 'dropna':
            df.loc[:, cols].dropna()
        else:
            df.loc[:, cols].apply(op)
        return df

    def joint_code(self, code_path, encoding='utf-8'):
        '''拼接代码文件'''
        try:
            f = open(os.path.join(self.code_files, code_path), 'r', encoding=encoding)
            self.generate += f.read() + '\n'
        except:
            f = open(os.path.join(self.code_files, code_path), 'r', encoding='gbk')
            self.generate += f.read() + '\n'

    def get_code(self):
        '''生成代码'''
        # 拼接导入的库
        self.joint_code('ImportPackages.py')
        self.generate += '\n' + MODEL_DICT[self.model_type][self.model_name] + '\n'

        # 拼接函数评估方法
        for method in self.evaluate_methods:
            self.joint_code(MODEL_DICT[method])

        # 拼接变量
        sklearn_model = MODEL_DICT[self.model_type][self.model_name].split(' ')[-1] + '()'
        self.generate += '''
FILE_PATH={}\n
FILE_PATH = np.unique(FILE_PATH)\n
FEATURES={}\n
TARGET='{}'\n
MODEL={}\n
model_name=str(MODEL)[0:-2]\n
        '''.format(self.dataset_name, self.features, self.target, sklearn_model)
        # 拼接主函数
        self.joint_code('Main.py')

        # 生成代码文件
        with open('generate.py', 'w', encoding='utf-8') as f:
            f.write(self.generate)
        f.close()


'''Flask'''
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
filename = []


@app.route('/')
def index():
    # return 'This is index!'
    return render_template('index.html')


@app.route('/<name>')
def user(name):
    return render_template(name)


@app.errorhandler(403)
def page_not_found(e):
    return render_template("404.html"), 403


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(405)
def page_not_found(e):
    return render_template("404.html"), 405


@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500


@app.errorhandler(503)
def page_not_found(e):
    return render_template("404.html"), 503


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return "upload GET Done"
    else:
        file = request.files.get('file')
        filename.append(file.filename)
        file.save(os.path.join('./Datasets/', filename[-1]))
        return 'Upload ' + filename[-1] + ' Done'


@app.route('/runmodel', methods=['GET', 'POST'])
def runmodel():
    if request.method == 'GET':
        return "runmodel GET Done"
    else:
        models = []
        models_arg = request.get_data()
        models_arg = str(models_arg)[2:-1].split('&')
        for i in models_arg:
            if i.split('=')[1][-1] != '0':
                models.append(i.split('=')[1])
        # print(models)
        evaluate = []
        alltype = []
        allname = []
        for i in models:
            if (i[0] > '0') & (i < '4'):
                alltype.append(i[0])
                allname.append(i)
            else:
                evaluate.append(i)

        if filename == []:
            return 'Please upload a file first!'
        if alltype == [] or allname == []:
            return 'Please choose a model!'
        if evaluate == []:
            return 'Please choose a evaluation method!'
        print(filename, alltype, allname, evaluate)

        f = open('./static/modelresult/sel_models.txt', 'w')
        for type, name in zip(alltype, allname):
            f.write(MODEL_DICT_name[type][name] + '*')
        f.close()
        if os.path.isfile('./static/modelresult/train_result.txt'):
            os.remove('./static/modelresult/train_result.txt')
        with open('./static/modelresult/train_result.txt', 'a+') as f:
            f.close()

        for type, name in zip(alltype, allname):
            myModel = SetModel(filename, 'target', 'features', type, name, evaluate)
            print(name, 'I am running...')
            myModel.get_code()
            os.system('python generate.py')
            print(name, 'Done')

        return 'Congratulations Model Done'


@app.route('/getSelectedModels', methods=['GET'])
def getSelectedModels():
    with open('./static/modelresult/sel_models.txt', 'r') as f:
        sel_models = f.read().split('*')[0:-1]
    data = {'Models': sel_models}
    res = json.dumps(data)
    print(res)
    return res


@app.route('/getTrainedResult', methods=['GET'])
def getTrainedResult():
    if request.method == 'GET':
        with open('./static/modelresult/sel_models.txt', 'r') as f:
            sel_models = f.read().split('*')[0:-1]
        print('number of models:', len(sel_models))
        data = {'TrainedResult': [], 'Finished': False}
        with open('./static/modelresult/train_result.txt', 'r') as f:
            _train_result = f.read().split('*')
        temp = []
        for i in range(len(_train_result)):
            if (i % 4 == 0) & (i != 0):
                data['TrainedResult'].append(temp)
                temp = []
                temp.append(_train_result[i])
            else:
                temp.append(_train_result[i])
        if i == len(sel_models) * 4:
            data['Finished'] = True
        res = json.dumps(data)
        print(res)
        return res
    else:
        return 'There is no need to POST!'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug='True')

# For the test
# myModel = SetModel(['flex.csv','punch.csv'], 'target', 'features', '1', 'decisiontree', ['ROCcurve','confusion'])
# myModel.get_code()
# import generate
