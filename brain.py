import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.optimizers import RMSprop
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 読み込み
brain_data = pd.read_csv("output_1.csv")

# 特徴量データと教師データに分離
X = brain_data.loc[:, ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]].values
y = brain_data.loc[:, "Name"].values

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# モデルの学習
estimator = SVC()
estimator.fit(X_train, y_train)

#モデルの保存
open('and.json', "w").write(estimator.to_json())

#学習データの保存
estimator.save_weights('and_weight.hsf5')

# モデルの評価
y_pred = estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print ("正解率 = ", accuracy)

# 未知データの分類
data = [[4.2, 3.1, 1.6, 0.5]]
X_pred = np.array(data)
y_pred = estimator.predict(X_pred)
print(y_pred)

