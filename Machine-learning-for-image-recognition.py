import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

import random

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.0001)

x,y = digits.data[:-1],digits.target[:-1]
clf.fit(x,y)

def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return 0
    elif reply[0] == 'n':
        return 1
    else:
        return yes_or_no("Please Enter (y/n) ")

while(True):
    n = random. randint(0,1000)
    print("Prediction of last:",clf.predict(digits.data[[n]]))
    plt.imshow(digits.images[n], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()
    
    if(yes_or_no('Try again')):
        break

print("done")
