import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#EDA : 

data=pd.read_csv('/content/kag_risk_factors_cervical_cancer.csv')
data.head
data.shape
data.info()
data.describe()
data.isnull().sum()
for feature in data.columns:
    data[feature].replace('?',np.nan,inplace=True )
    data[feature].fillna(value=0,inplace=True)
for feature in data.columns:
    data[feature].replace(0,data[feature].median(),inplace=True)
data.head()
  
#Data visualization
  
sns.distplot(data['Age'])
df=pd.DataFrame(data[['Hinselmann','Schiller','Citology','Biopsy']])
df.head(10)

for features in df.columns:
    s=df.copy()
    sns.countplot(x=s[features])
    plt.xlabel(features)
    plt.title(features)
    plt.show()
  
for features in df.columns:
    s=df.copy()
    sns.barplot(x=s[features],y=data['Age'])
    plt.xlabel(features)
    plt.ylabel("Age")
    plt.title(features)
    plt.show()
  
sns.heatmap(df.corr(),annot=True)

sns.heatmap(data.corr())

df['count']=df['Hinselmann']+df['Schiller']+df['Citology']+df['Biopsy']
df['count'].value_counts()
df['result']=np.where(df['count']>0,1,df['count'])
df['result'].value_counts()
X=data.drop(columns=['Hinselmann','Schiller','Citology','Biopsy'],axis=1)
y=df['result']

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_feature=scaler.fit_transform(X)
X_scaled=df
y=df['result']

#Split the data into train/test.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)
X_train.shape,y_train.shape
X_test.shape,y_test.shape

#Using Logistic Regression

from sklearn.linear_model import LogisticRegression
l_r=LogisticRegression()
model=l_r.fit(X_train,y_train)
pred=model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Confusion Matrix:\n ', confusion_matrix(y_test,pred))
print('Classification Report:\n ',classification_report(y_test,pred))

#Logistic Regression Using Kfold Cross validation

#In this we are going to predict using same model but with kfold Cross validation and obtain the accurcay.

from sklearn.model_selection import KFold,cross_val_score
kfold=KFold(n_splits=10,shuffle=True,random_state=21)
model=LogisticRegression()
scores=cross_val_score(model,X,y,scoring='accuracy',cv=kfold,n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#KNN with CV

#In this we are going to use K Nearest Neighbor and find the best K value using Grid Search CV and also using Kfold cross validation.

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_jobs=-1)
knn_neighbors={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
from sklearn.model_selection import GridSearchCV
classifier=GridSearchCV(knn,param_grid=knn_neighbors,cv=kfold,verbose=0).fit(X_train,y_train)
classifier.best_params_
best_grid=classifier.best_estimator_
best_grid
predict=best_grid.predict(X_test)
print('Confusion Matrix:\n ',confusion_matrix(y_test,predict))
print('Classification Report:\n ',classification_report(y_test,predict))

#Using Descision Tree with Kfold CV

from sklearn.tree import DecisionTreeClassifier
df_tree=DecisionTreeClassifier(random_state=0)
df_model=df_tree.fit(X_train,y_train)
df_pred=df_model.predict(X_test)
print('Confusion Matrix:\n ',confusion_matrix(y_test,df_pred))
print('Classification Report:\n ',classification_report(y_test,df_pred))

#Using kfold Cross validation with descision tree model.

score=cross_val_score(df_model,X,y,scoring='accuracy',cv=kfold,n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(score), np.std(score)))

#Using Random Forest

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf_model = rf.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)
print('Confusion Matrix:\n ',confusion_matrix(y_test,rf_pred))
print('Classification Report:\n ',classification_report(y_test,rf_pred))

#Using Kfold CV in random forest model

s=cross_val_score(rf_model,X,y,scoring='accuracy',cv=kfold,n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(s), np.std(s)))

#prediction using console

a=int(input("enter your age:"))
b=int(input("enter no.of pregnancies:"))
c=int(input("enter no.of.partners:"))
d=int(input("enter your first sex initalization age:"))
e=int(input("enter your HPV type:"))
f=int(input("enter smokes per day :"))
y=model.predict([[a,b,c,d,e,f]])
if(y<=0.5):
  print("THERE IS NO CHANCE OF GETTING CERVICAL CANCER")
elif(0.5>y<0.7):
  print("STAGE I!! BUT CONSULT THE DOCTOR")
elif(0.7>y<0.9):
  print("STAGE II!!BUT CONSULT THE DOCTOR")
else :
  print("INVALID")

#for GUI 

pip install gradio
import gradio as gr
def cervical(Age,No_of_partners,Initialization_age,Num_of_pregnancies,Smokes,DxHPV):
  x=np.array([Age,No_of_partners,Initialization_age,Num_of_pregnancies,Smokes,DxHPV])
  ypred=rf_model.predict(x.reshape(1,-1))
  if ypred==1:
    return "THERE IS CHANCE OF GETTING CERVICAL CANCER!!CONSULT THE DOCTOR"
  else:
    return "LESS POSSIBLE OF GETTING CERVICAL CANCER"

outputs=gr.outputs.Textbox()
app=gr.Interface(fn=cervical,inputs=['number','number','number','number','number','number'],outputs=outputs,description="Cervical cancer prediction")

app.launch()

#to check the mathematical evaluation

X=X_train[:10]
Y=y_train[:10]

from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,Y)
tree.plot_tree(clf)

import graphviz
dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
graph.render("CERVICAL CANCER PREDICTION")

