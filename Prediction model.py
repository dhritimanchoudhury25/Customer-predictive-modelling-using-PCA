# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

#Construct Data For Training the model
df=pd.read_csv()

outcome=df[['smart_id','outcome']]
outcome=outcome.drop_duplicates()
piv=pd.pivot_table(data=df,values=['gmv','dau','dcu','rto','u_coupon'],index=['smart_id'],columns=['week'],fill_value=0)
pivot=pd.merge(piv,outcome,how='left',left_on='smart_id',right_on='smart_id')
file_name = "E:\Data_Predictive_Modelling_Shopx_Plus\Shopx_Plus_Data\Pivot_Data_Training" + ".csv"
pivot.to_csv(file_name, encoding='utf-8', index=False)

#Data to get prediction output
df=pd.read_csv('E:\Data_Predictive_Modelling_Shopx_Plus\Shopx_Plus_Data\Data_source_Actual.csv')
outcome=df[['smart_id','outcome']]
outcome=outcome.drop_duplicates()
piv=pd.pivot_table(data=df,values=['gmv','dau','dcu','rto','u_coupon'],index=['smart_id'],columns=['week'],fill_value=0)
pivot=pd.merge(piv,outcome,how='left',left_on='smart_id',right_on='smart_id')
file_name = "E:\Data_Predictive_Modelling_Shopx_Plus\Shopx_Plus_Data\Pivot_Data_Actual" + ".csv"

pivot.to_csv(file_name, encoding='utf-8', index=False)


# Importing the dataset for training
file_name = "E:\Data_Predictive_Modelling_Shopx_Plus\Shopx_Plus_Data\Pivot_Data_Training" + ".csv"
dataset = pd.read_csv(file_name)
X = dataset.iloc[:, 1:70].values
y = dataset.iloc[:, 71].values

#get data for prediction
file_name = "E:\Data_Predictive_Modelling_Shopx_Plus\Shopx_Plus_Data\Pivot_Data_Actual" + ".csv"
dataset = pd.read_csv(file_name)
z=dataset.iloc[:, 1:70].values

# Splitting the dataset into the Training set and Test set



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Perfom data preprocessing and standardization(Standardize the data through mean and variance)
#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#apply the compression n_value to 5
pca = PCA(n_components=5)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)
# print(z)
z_pca=pca.fit_transform(z)



# Fitting XGBoost to the training data
import xgboost as xgb
my_model = xgb.XGBClassifier()
my_model.fit(X_train, y_train)
my_model.fit(X_train, y_train)



#predict test set and get accuracy
y_pred=my_model.predict(X_test)

# Predicting the Data set results
z_pred = my_model.predict(z_pca)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


print("Accuracy Percentage\n")
print(((cm[0][0]+cm[1][1])/np.sum(cm))*100)


dataset['Predicted_Outcome']=z_pred

print("Writing the predictated data to file Predictive_Data_Mapped ")
file_name = "E:\Data_Predictive_Modelling_Shopx_Plus\Shopx_Plus_Data\Predictive_Data_Mapped" + ".csv"


dataset.to_csv(file_name, encoding='utf-8', index=False)

print("Data write complete")

