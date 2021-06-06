import numpy as np
from sklearn.decomposition import PCA
import joblib

x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=1)
newx = pca.fit_transform(x)  # 等价于pca.fit(X) pca.transform(X)

joblib.dump(pca, './pca.m')  # 将模型保存到pca.m文件中
x_test1 = pca.transform(x)  # 将模型运用到新数据中，此处就用x做测试
print(x_test1)
pca2 = joblib.load('./pca.m')  # 读入模型pca.m
x_test2 = pca2.transform(x)
print(x_test2)

# invx = pca.inverse_transform(newx)  # 将降维后的数据转换成原始数据
# testx = pca.transform(x)  # 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。

print(pca.explained_variance_ratio_)  # 返回所保留的n个成分各自的方差百分比
