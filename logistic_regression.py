import os
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

train=pd.read_csv('C:/data_science/janatahack/train.csv')
test=pd.read_csv('C:/data_science/janatahack/test.csv')
train_x=train.drop(['ID','default_payment_next_month'],axis=1)
test_x=test.drop('ID',axis=1)
idval=test['ID']
train_y=train['default_payment_next_month']
train.drop('ID',axis=1)
#x_train, x_validat, y_train, y_validat = train_test_split(train_x,train_y, random_state = 0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(solver='lbfgs',max_iter=300, multi_class='auto') 
logclass=classifier.fit(train_x, train_y) 
predict=logclass.predict(test_x)
submission = pd.DataFrame({ 'ID': idval, 'default_payment_next_month': predict })
submission.to_csv("submission.csv", index=False)
#fpr, tpr, thresholds = roc_curve(y_validat,probs)