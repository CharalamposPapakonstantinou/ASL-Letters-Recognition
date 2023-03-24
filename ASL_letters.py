#CREATE DATA

import cv2
import mediapipe as mp
import imutils
import pandas as pd
import numpy as np


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)
    return results


def draw_hand(img, results,id):
    L = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            HL=list(enumerate(hand.landmark))
            for p in id:
                lm=HL[p][1]
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(p, cx, cy)
                L.extend([cx,cy])

                cv2.circle(img, (cx, cy), 3, (55, 255, 0), cv2.FILLED)
                # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # print(L)
    return img,L




G=25
cap = cv2.VideoCapture(0) # NOTE!!!!: TO RUN THIS, RUN PYCHARM THROUGH ANACONDA!!!!
id=range(0,21)#[0,2,4,5,8,9,12,13,16,17,20]
pini=[0]*len(id)*2
p=pd.DataFrame(pini).transpose()
counter=0
while True:
    # Taking the input
    success, image = cap.read()
    image = imutils.resize(image, width=500, height=500)
    results = process_image(image)
    im,L=draw_hand(image, results,id)

    if L!=[]:
        p.loc[len(p)] = L[:len(id)*2]
        counter = counter + 1



    # Displaying the output
    image=cv2.flip(image,1)
    cv2.imshow("Hand tracker", image)

    # Program terminates when q key is pressed
    print(counter)
    if cv2.waitKey(1) == ord('q') or counter>500:
        p = p.tail(-1)
        p["Gesture"] = [G] * len(p)
        p.to_excel('/Users/charalamposp/Desktop/gestures/handgesture'+str(G)+'.xlsx', index=False)
        cap.release()
        cv2.destroyAllWindows()




##  TRAIN THE MODEL

import pandas as pd
import numpy as np
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,accuracy_score,ConfusionMatrixDisplay,confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import joblib

result_pd = pd.DataFrame()
N=25 #how many excel files/gestures
for G in range(0,N+1):
    df = pd.read_excel('/Users/charalamposp/Desktop/gestures/handgesture' + str(G) + '.xlsx')
    result_pd = pd.concat([result_pd, df])
print(result_pd)


nndata=np.array(result_pd)


x0=nndata[:,0]
y0=nndata[:,1]
data_aug=nndata
for i in range(2,nndata.shape[1]-1,2):
    xtemp=nndata[:,i]
    ytemp=nndata[:,i+1]
    dist=np.round(np.sqrt(np.square(xtemp-x0)+np.square(ytemp-y0))).astype(int)
    data_aug = np.insert(data_aug, -1, dist, 1)



for sf in range(0,360):
    np.random.shuffle(data_aug)

R,C=data_aug.shape



x=data_aug[:,0:C-1]
y=data_aug[:,C-1]
x, x_test, y, y_test = train_test_split(x, y, test_size=0.3, random_state=41,shuffle=True)

# scaler = StandardScaler()
# scaler.fit(x)
# x=scaler.transform(x)
# scaler.fit(x_test)
# x_test=scaler.transform(x_test)

clf = RandomForestClassifier(n_estimators=200,n_jobs=-1, random_state=0,criterion='entropy',oob_score=True,max_features=7)
clf.fit(x, y)
pred=clf.predict(x_test)


mae=mean_absolute_error(y_test,pred)
print('Mean absolute error = ', mae)

CM=confusion_matrix(y_test,pred)
print(CM)
print(accuracy_score(y_test,pred))

disp = ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=clf.classes_)
disp.plot()

precision_recall_fscore_support(y_test, pred, average='macro')


# save the model
joblib.dump(clf, "RFC_gestures.joblib")



# for xt in  x_test:
#     #one sample prediciton
#     XT=np.reshape(xt, (1,len(xt)))
#     pred=clf.predict(XT)
#     # print(int(pred))


## VIDEO & PREDICTIONS

import cv2
import mediapipe as mp
import imutils
import pandas as pd
import numpy as np
import os
from pync import Notifier
import pyautogui
import time
import joblib
import string



# load
clf = joblib.load("RFC_gestures.joblib")


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)
    return results


def draw_hand(img, results,id):
    L = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            HL=list(enumerate(hand.landmark))
            for p in id:
                lm=HL[p][1]
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(p, cx, cy)
                L.extend([cx,cy])

                cv2.circle(img, (cx, cy), 3, (55, 255, 0), cv2.FILLED)
                # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # print(L)
    return img,L


def phrase(lst):
    result = ""

    for i in range(len(lst)):
        if lst[i] != "":
            if i < len(lst) - 1 and lst[i + 1] != "":
                result += lst[i]
            else:
                result += lst[i] + " "

    phrase=result.strip()
    return phrase


def augment(XT):
    XT=XT.ravel()
    x0 = XT[0]
    y0 = XT[1]
    XT_aug = XT
    for i in range(2, XT.shape[0] - 1, 2):
        xtemp = XT[i]
        ytemp = XT[i + 1]
        dist = np.sqrt(np.square(xtemp - x0) + np.square(ytemp - y0)).astype(int)
        XT_aug = np.append(XT_aug, dist)
    return XT_aug.reshape(-1,XT_aug.shape[0])

cap = cv2.VideoCapture(0) # NOTE!!!!: TO RUN THIS, RUN PYCHARM THROUGH ANACONDA!!!!
id=range(0,21)#[0,2,4,5,8,9,12,13,16,17,20]
pini=[0]*len(id)*2
p=pd.DataFrame(pini).transpose()


templist=[]
signal=['']
winmax=0
lex = string.ascii_lowercase
prev_pred=''
while True:
    #input
    success, image = cap.read()
    image = imutils.resize(image, width=500, height=500)
    results = process_image(image)
    im,L=draw_hand(image, results,id)

    if L!=[]:
        p.loc[len(p)] = L[:len(id)*2]

        # one sample prediciton
        x=list(p.iloc[-1])
        XT=np.reshape(x, (1,len(x)))
        XT_aug=augment(XT)

        pred=clf.predict(XT_aug)
        cur_pred=int(pred)

        is_moving=np.mean(abs(p.iloc[-1] - p.iloc[-2])) > 2


        if (cur_pred!=prev_pred or signal[-1]=='') and is_moving==False:
            letter=lex[int(pred)]
            # print(pred,end='')
            signal.append(letter)
            print(phrase(signal))

            prev_pred=cur_pred

    else:
        if signal[-1]!='':
            # print(' ', end='')
            signal.append('') #fill with empty, gia na mh thymatai to prohgoymeno poy mporei na egine prin poly wra



    # if len(signal) >= 50: signal = []




    # Displaying the output
    image=cv2.flip(image,1)
    cv2.putText(image, phrase(signal), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0))

    cv2.imshow("Hand tracker", image)


    # Program terminates when q key is pressed
    if cv2.waitKey(1) == ord('q'):
        p = p.tail(-1)
        cap.release()
        cv2.destroyAllWindows()


