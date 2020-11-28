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
    print(data)
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("letter: " + str(label))
    #plt.axis("off")
    plt.show()
sample_id = 10

#Vẽ một hình ở row thứ sample_id, truyền vào feature, và label. 
sample_id = 15
plot_digit(X_train[sample_id], y_train[sample_id])

#============================= SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

if 0:
   sgd_clf = sgd_clf.fit(X_train, y_train)
   joblib.dump(sgd_clf, 'saved_var/sgd_clf')
else:
   sgd_clf = joblib.load('saved_var/sgd_clf')

#Đầu tiên xem danh sách class:
sgd_clf.classes_
    


# Try prediction
# Dự đoán thử.
sample_id = 156
print("SGD predict:")
print(sgd_clf.predict([X_train[sample_id]]))
print("Actual result:")
y_train[sample_id]
# To see scores from classifers
#Để xem điểm của classifiers.


#Tính score
#Lấy maximum của score.
sample_scores = sgd_clf.decision_function([X_train[sample_id]])
class_with_max_score = np.argmax(sample_scores)

# In[6]: EVALUATE CLASSIFIERS
from sklearn.model_selection import cross_val_score
# 6.1. SGDClassifier  
# Warning: takes time for new run! 
#Tính accuracy, dùng cross_val_score, chỉ định scoring = "accuracy"

print('starting calculating SDG accuracy...')
if 0:
    sgd_acc = cross_val_score(sgd_clf, X_train, y_train, cv=3, n_jobs=-1, scoring="accuracy")
    joblib.dump(sgd_acc,'saved_var/sgd_acc')
else:
    sgd_acc = joblib.load('saved_var/sgd_acc')
print('complete... this is result of SGD:')
print(sgd_acc)
print('-------------------------------------------------')

none = 1