# In[0]: IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from emnist import list_datasets
list_datasets()

#['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
#Classes:
#classes 0 -> 9: số 0 đến 9.
#10: A
#11: B
#12: c, C
#13: D
#14: E
#15: F
#16: G
#17: H
#18: i, I
#19: j, J
#20: k, K
#21: l, L
#22: m, M
#23: N 
#24: o, O
#25: p, P 
#26: Q
#27: R
#28: s, S
#29: T 
#30: u, U
#31: v, V 
#32: w, W 
#33: x, X
#34: y, Y 
#35: z, Z
#36: a
#37: b
#38: d
#39: e
#40: f 
#41: g
#42: h
#43: n
#44: q
#45: r
#46: t

#Get train test.
from emnist import extract_training_samples
X_train, y_train = extract_training_samples('bymerge')

#Get test set.
from emnist import extract_test_samples
X_test, y_test = extract_test_samples('bymerge')

#In ra kiểm tra số lượng image train và test.
#X_train, X_test là samples, y_train, y_test là nhãn.
print("x_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("x_test shape:", X_test.shape, "y_test shape:", y_test.shape)

#Đổi matrix 28*28 pixel, thành một mảng 1 chiều 784 phần tử.
X_train = X_train.reshape(697932,784)
X_test = X_test.reshape(116323,784)

# 1.3. Plot a digit image
#Function để vẽ hình ra
#Chỉ cần đưa mảng 2 chiều của một hình, nó sẽ vẽ được hình hoàn chỉnh.
#Trước khi vẽ, cần kéo mảng 1 chiều thành mảng 2 chiều 28*28 pixel.
def plot_digit(data, label = 'unspecified'):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("letter: " + str(label))
    #plt.axis("off")
    plt.show()
sample_id = 10

#Vẽ một hình ở row thứ sample_id, truyền vào feature, và label. 
sample_id = 15
#plot_digit(X_train[sample_id], y_train[sample_id])

#============================= RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
from sklearn.model_selection import cross_val_predict

#Fit với data chưa được scale
#if 0:
#    rf_clf = rf_clf.fit(X_train, y_train)
#    joblib.dump(rf_clf, 'saved_var/rf_clf')
#else:
#    rf_clf = joblib.load('saved_var/rf_clf')
#print('học xong RandomForestClassifier')

#if 0:
#   y_probas_rf = cross_val_predict(rf_clf, X_train, y_train, cv=3, n_jobs=-1, method ="predict_proba")
#   joblib.dump(y_probas_rf, 'saved_var/y_probas_forest')
#else:
#   y_probas_rf = joblib.load('saved_var/y_probas_forest')
#print('xong predict_proba y_probas_rf')

#############

############
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# 7.1. SGDClassifier (benefited from feature scaling)
# Warning: takes time for new run! 

#Chạy lại RF, nhưng dữ liệu đầu vào đã được scale.

#Fit model

print('Bắt đầu fit với dữ liệu được scale')

if 0:
    rf_clf_after_scale = rf_clf_after_scale.fit(X_train_scaled, y_train)
    joblib.dump(rf_clf_after_scale, 'saved_var/rf_clf_after_scale')
else:
    rf_clf_after_scale = joblib.load('saved_var/rf_clf_after_scale')
print('học xong RandomForestClassifier')


if 1:
    rf_acc_after_scaling = cross_val_score(rf_clf_after_scale, X_train_scaled, y_train, cv=3, scoring="accuracy")
    joblib.dump(rf_acc_after_scaling,'saved_var/rf_acc_after_scaling')
else:
    rf_acc_after_scaling = joblib.load('saved_var/rf_acc_after_scaling')
#

#
#if 1:
#    rf_acc_after_scaling_test_set = cross_val_score(rf_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
#    joblib.dump(rf_acc_after_scaling_test_set,'saved_var/rf_acc_after_scaling_test_set')
#else:
#    rf_acc_after_scaling_test_set = joblib.load('saved_var/rf_acc_after_scaling_test_set')





## [5] Try Predict
#print("Random forest predict:")
#sample_id = 155
#print(rf_clf.predict([X_train[sample_id]]))

#print("Actual result:")
#y_train[sample_id]

## In[6]: EVALUATE CLASSIFIERS
#from sklearn.model_selection import cross_val_score

## 6.2. RandomForestClassifier  
##Tính accuracy, dùng cross_val_score, chỉ định scoring = "accuracy"
## Warning: takes time for new run! 

#print('start calculate accuracy score...');
#if 0:
#    forest_acc = cross_val_score(rf_clf, X_train, y_train, cv=3, n_jobs=-1, scoring="accuracy")
#    joblib.dump(forest_acc,'saved_var/forest_acc')
#else:
#    forest_acc = joblib.load('saved_var/forest_acc')


#print('complete... this is result of Random forest:')
#print(forest_acc)
#none = 1
