# In[0]: IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


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


#============================= RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

from sklearn import svm

from sklearn.model_selection import cross_val_predict
from PIL import Image
from matplotlib.pyplot import imread
from skimage.color import rgb2gray, rgba2rgb
class RandomForestPredict:
    #rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = joblib.load('saved_var/rf_clf')
    #svm_clf = joblib.load('saved_var/svm_clf_best_parameter')
    print('học xong svm_clf_best_parameter')

    def predict(self, image_name):
        self.image = imread(image_name)
        img = rgb2gray(rgba2rgb(self.image));

        print('Image shape:')
        img_arr = img.reshape(784)

        print("=======================================")
        print("=======================================")
        for i in range(0,784):
            a = 256 * img_arr[i]
            img_arr[i] = int(a)
            img_arr[i] = 255 - img_arr[i]

        #print(img_arr)
        #
        
        #i = img_arr.reshape(28, 28)
        #plt.imshow(i, cmap = mpl.cm.binary)
        #plt.title("letter: ")
        #plt.show()
        #
        results = self.svm_clf.predict_proba([img_arr])[0]
        print(results)
        results = np.array(results)
        first_max = -1
        fist_index = -1
        second_max = -1
        second_index = -1
        third_max = -1
        third_index = -1
        fourth_max = -1
        fourth_index = -1
        fifth_max = -1
        fifth_index = -1

        for i in range(0, results.size):
            results[i] = results[i]*100
            if(results[i] >= first_max):
                first_max = results[i]
                first_index = self.numbers_to_strings(i)
                continue
            if(results[i] >= second_max):
                second_max = results[i]
                second_index = self.numbers_to_strings(i)
                continue
            if(results[i] >= third_max):
                third_max = results[i]
                third_index = self.numbers_to_strings(i)
                continue
            if(results[i] >= fourth_max):
                fourth_max = results[i]
                fourth_index = self.numbers_to_strings(i)
                continue
            if(results[i] >= fifth_max):
                fifth_max = results[i]
                fifth_index = self.numbers_to_strings(i)


        kq = 'results:\n' + '- ' + str(first_index) + ': ' + str(round(first_max, 2)) + '%\n' + '- ' + str(second_index) + ': ' + str(round(second_max, 2)) + '%\n' + '- ' + str(third_index) + ': ' + str(round(third_max,2)) + '%\n'  + '- ' + str(fourth_index) + ': ' + str(round(fourth_max, 2)) + '%\n'  + '- ' + str(fifth_index) + ': ' + str(round(fifth_max, 2)) + '%\n'
        print(kq)
        return kq

    
    def numbers_to_strings(self,char): 
        switcher = { 
            0: '0', 
            1: '1', 
            2: '2', 
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: 'A',
            11:'B',
            12: 'c, C',
            13:'D',
            14:'E',
            15:'F',
            16:'G',
            17:'H',
            18:'i, I',
            19:'j, J',
            20: 'k, K',
            21: 'l, L',
            22: 'm, M',
            23: 'N',
            24:'o, O',
            25:'p, P',
            26:'Q',
            27:'R',
            28:'s, S',
            29:'T',
            30:'u, U',
            31:'v, V',
            32:'w, W', 
            33:'x, X',
            34:'y, Y',
            35:'z, Z',
            36:'a',
            37:'b',
            38:'d',
            39:'e',
            40:'f',
            41:'g',
            42:'h',
            43:'n',
            44:'q',
            45:'r',
            46:'t'
        }
        return switcher.get(char, "nothing")

    def predict2():
        image = Image.open('test.png')
        image.thumbnail((28, 28))

        self.image = imread(image_name)
        img = rgb2gray(rgba2rgb(self.image));

        print('Image shape:')
        img_arr = img.reshape(784)

        for i in range(0,784):
            img_arr[i] = 255 - int(img_arr[i])
        print(img_arr)
        #
        
        #i = img_arr.reshape(28, 28)
        #plt.imshow(i, cmap = mpl.cm.binary)
        #plt.title("letter: ")
        #plt.show()
        #

        print(self.svm_clf.predict([img_arr]))
