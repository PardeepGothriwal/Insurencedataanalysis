#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd C:\Users\DELL\Desktop


# In[2]:


pwd


# # Insurance Data
# 
# ## Author = PARDEEP
# ## Update = 14/03/2024
# 
# 
# 
# ### Data Exploration and Cleaning
# 
# 1. **Load the Data**
#         - Load the insurance dataset into a pandas DataFrame
# 2. **Initial Inspection**
#     - Display the first few rows of the Dataset
#     - Check for missing values and handle them if necessary
# 
# ### Summary Statistics
# 
# 3. **Summary Statistics**
#     - Display summary statistics for the numerical columns
# 4. **Distribution of variables**
#     - Explore unique values in categorical columns by value counts
# 

# ## Import Packages

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as py
py.init_notebook_mode(connected=True)

from math import sqrt
from pprint import pprint
from IPython.display import display
from scipy.spatial import ConvexHull
from matplotlib.pyplot import figure

from sklearn.metrics import r2_score
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split,RandomizedSearchCV


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# In[4]:


df = pd.read_csv("insurance.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df


# In[8]:


#Check for the missing values
missing_values = df.isnull().sum()
print("The Number of the missing values:\n", missing_values)

print("*"*30)
#Checking for the statistical summary
print("Statistical Check:\n")
print(df.describe().T)
print("*"*30)

#Checking for the uniques values in the category columns
age_unique = df["age"].unique()
sex_unique = df["sex"].unique()
children_unique = df["children"].unique()
region_unique = df["region"].unique()

print("Age unique: ", age_unique)
print("Sex unique: ", sex_unique)
print("Children unique: ", children_unique)
print("Region unique: ", region_unique)


# Checking for duplicate values

# In[9]:


df.duplicated().sum()


# Dropping duplicate values in the dataframe

# In[10]:


df.drop_duplicates(inplace=True)


# Double check for duplicate values to make sure that they are removed

# In[11]:


df.duplicated().sum()


# In[12]:


#Checking for data types

df.info()


# In[13]:


#Checking for the sample datatype
first_row = df.head()

for column in first_row.columns:
    column_values = first_row[column]
    value = column_values.values[0]
    #value_type = type(value).__name__
    print(f"column_values: {column}, value: {value}, Type: {type(value).__name__} \n")


# In[14]:


first_row.columns


# In[15]:


for column in first_row.columns:
    column_values = first_row[column]
    value = column_values[0]
    value_type =type(value).__name__
    #print(value)


# In[16]:


type(19)


# In[17]:


#Checking the unique values to present normal distribution
df["children"].value_counts().plot(kind = "bar", title = "Children Distibution")
plt.grid()
plt.ylabel("Amount")


# In[18]:


df["sex"].value_counts().plot(kind = "bar", title = "Sex Distibution")
plt.grid()
plt.ylabel("Amount")


# In[19]:


df["smoker"].value_counts().plot(kind = "bar", title = "Smoker Distibution", color = "r")
plt.grid()
plt.ylabel("Amount")


# In[20]:


df["region"].value_counts().plot(kind = "bar", title = "Region Distibution", color = "g")
plt.grid()
plt.ylabel("Amount")


# In[21]:


#Understanding BMI visualisation
sns.set(style="whitegrid")
plt.figure(figsize = (10,8))
sns.kdeplot(df["charges"],fill =True, palette="viridis",linewidth =3)
plt.title("CHARGES Distribution")
plt.xlabel("BMI Value")
plt.savefig("Charges_plot.png", dpi = 500)
plt.show()


# In[22]:


from scipy import stats  



transformed_charges, lambda_value = stats.boxcox(df["charges"])
print(lambda_value)

plt.figure(figsize=(10, 8))
sns.kdeplot(transformed_charges, fill=True, palette="viridis", linewidth=3)
plt.title("Transformed CHARGES Distribution (Box-Cox)")
plt.xlabel("Transformed Values")
plt.savefig("Transformed_Charges_plot.png", dpi=500)
plt.show()


# In[23]:


transformed_charges = np.sqrt(df["charges"])


plt.figure(figsize=(10, 8))
sns.kdeplot(transformed_charges, fill=True, palette="viridis", linewidth=3)
plt.title("Transformed CHARGES Distribution (Square Root)")
plt.xlabel("Transformed Values")
plt.savefig("Transformed_Charges_plot_sqrt.png", dpi=500)
plt.show()


# In[24]:


transformed_charges = np.log(df["charges"])


plt.figure(figsize=(10, 8))
sns.kdeplot(transformed_charges, fill=True, palette="viridis", linewidth=3)
plt.title("Transformed CHARGES Distribution (Logarithm)")
plt.xlabel("Transformed Values")
plt.savefig("Transformed_Charges_plot_log.png", dpi=500)
plt.show()


# ## Correlation application

# In[25]:


df["sex"]=df["sex"].replace({"male":0, "female":1})
df["smoker"]=df["smoker"].replace({"yes":1, "no":0})
df["region"].unique()


# In[26]:


df.drop(["region"],axis=1, inplace=True )
correlation_matrix = df.corr(method= "pearson")


print("Correlation Matrix:")
#print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("Correlation_Matrix.png", dpi=500)
plt.show()


# ## Visualising plots

# In[27]:


sns.violinplot(x="smoker",y="charges",data=df)


# In[28]:


sns.violinplot(x="smoker",y="bmi",data=df)


# In[29]:


sns.violinplot(x="smoker",y="age",data=df)


# In[30]:


sns.scatterplot(x = 'age' , y = 'charges' , hue = 'smoker' , data = df , palette = {1: 'red' , 0: 'blue'})
plt.title("scatter plot for the age and charges and  smoker in relation")
plt.show()


# In[31]:


df.head(5)


# In[ ]:




