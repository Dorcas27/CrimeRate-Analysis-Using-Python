#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Calling Dataset
crimerate = pd.read_csv("BaltimoreCrimerate.csv")


# In[5]:


crimerate.head()


# In[294]:


# Exploratory Data Analysis


# In[8]:


crimerate.shape


# In[10]:


crimerate.dtypes


# In[12]:


#finding total number of missing values
print(crimerate.isnull().sum())


# In[14]:


print(crimerate.describe())


# In[16]:


#finding unique values in specific categorical columns
print(crimerate['CrimeTime'].value_counts())


# In[18]:


print(crimerate['Neighborhood'].value_counts())


# In[20]:


print(crimerate['District'].value_counts())


# In[22]:


print(crimerate['Weapon'].value_counts())


# In[24]:


print(crimerate['Inside/Outside'].value_counts())


# In[26]:


#Using Replace function
crimerate['Inside/Outside'] = crimerate['Inside/Outside'].replace({'Inside': 'I','Outside': 'O'})


# In[28]:


print(crimerate.head())


# In[30]:


# Separating Date - goal is to have year, month and date in separate columns


# In[32]:


crimerate['CrimeDate'].str.split(pat = '/').str[2]


# In[34]:


crimerate['CrimeYear'] = crimerate['CrimeDate'].str.split(pat = '/').str[2]


# In[36]:


crimerate['CrimeDate'].str.split(pat = '/').str[1]


# In[38]:


crimerate['CrimeDay'] = crimerate['CrimeDate'].str.split(pat = '/').str[1]


# In[40]:


crimerate['CrimeDate'].str.split(pat = '/').str[0]


# In[42]:


crimerate['CrimeMonth'] = crimerate['CrimeDate'].str.split(pat = '/').str[0]


# In[44]:


crimerate


# In[46]:


#Drop CrimeDate column
crimerate.drop(['CrimeDate'], axis =1)


# In[48]:


crimerate['CrimeYear'].astype('int')


# In[50]:


crimerate['CrimeMonth'].astype('int')


# In[52]:


crimerate['CrimeDay'].astype('int')


# In[54]:


crimerate = crimerate.drop(['CrimeDate'], axis =1)


# In[56]:


crimerate


# In[58]:


crimerate['CrimeYear'] = crimerate['CrimeYear'].astype('int')


# In[60]:


crimerate['CrimeMonth'] = crimerate['CrimeMonth'].astype('int')


# In[62]:


crimerate['CrimeDay'] = crimerate['CrimeDay'].astype('int')


# In[64]:


# extracting hours, minutes and seconds
crimerate['CrimeTime'].str.split(pat = ':').str[0]


# In[66]:


crimerate['CrimeTime'].str.split(pat = ':').str[1]


# In[68]:


crimerate['CrimeTime'].str.split(pat = ':').str[2]


# In[70]:


crimerate['CrimeHours'] = crimerate['CrimeTime'].str.split(pat = ':').str[0]


# In[72]:


crimerate['CrimeMinutes'] = crimerate['CrimeTime'].str.split(pat = ':').str[1]


# In[74]:


crimerate['CrimeSeconds'] = crimerate['CrimeTime'].str.split(pat = ':').str[2]


# In[76]:


# drop CrimeTime
crimerate.drop(['CrimeTime'],axis =1)


# In[78]:


crimerate['CrimeHours'].astype('int')


# In[80]:


crimerate['CrimeHours'] = crimerate['CrimeHours'].astype('int')


# In[82]:


crimerate['CrimeMinutes'].astype('int')


# In[84]:


crimerate['CrimeMinutes']= crimerate['CrimeMinutes'].astype('int')


# In[86]:


crimerate['CrimeSeconds'].astype('int')


# In[88]:


crimerate['CrimeSeconds']= crimerate['CrimeSeconds'].astype('int')


# In[90]:


crimerate = crimerate.drop(['CrimeTime'],axis =1)


# In[92]:


crimerate


# In[94]:


# imputing missing values in categorical and numeric columns


# In[96]:


from sklearn.impute import SimpleImputer


# In[128]:


missing_categoricalcolumns = crimerate[['Location','Weapon','District','Premise']]


# In[130]:


imputer = SimpleImputer(strategy = 'constant',fill_value='missing')


# In[132]:


imputer.fit_transform(missing_categoricalcolumns)


# In[134]:


missing_numericalcolumns = crimerate[['Post','Latitude','Longitude','Location 1']]


# In[136]:


imputer_numeric = SimpleImputer(strategy = 'most_frequent')


# In[138]:


imputer_numeric.fit_transform(missing_numericalcolumns)


# In[ ]:





# In[140]:


# Visualizing CrimeRate Trend over the years by grouping and counting occurrences
yearly_trend = crimerate.groupby('CrimeYear').size()


# In[146]:


plt.figure(figsize=(10, 6))
yearly_trend.plot(kind='line', marker='o', color='b', linestyle='-', linewidth=2)
# Adding title and labels
plt.title('Yearly Crime Trends', fontsize=16)
plt.xlabel('Crime Year', fontsize=12)
plt.ylabel('Crime Count', fontsize=12)
plt.grid(alpha=0.3) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# In[148]:


# Visualizing CrimeRate Trend over the years by grouping and counting occurrences
monthly_trend = crimerate.groupby('CrimeMonth').size()


# In[152]:


plt.figure(figsize=(10, 6))
monthly_trend.plot(kind='bar', color='skyblue', width=0.8)
# Adding title and labels
plt.title('Monthly Crime Distribution', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Crime Count', fontsize=12)
plt.xticks(ticks=range(12), labels=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], rotation=45, fontsize=10)

plt.yticks(fontsize=10)
plt.grid(axis='y', alpha=0.3) 
plt.tight_layout()
plt.show()


# In[154]:


#Visualizing Location-base CrimeRate Analysis


# In[158]:


# Visualizing Crimerate Distribution by District
district_crime = crimerate['District'].value_counts(normalize=True) * 100
# Plotting Crime Distribution by District
plt.figure(figsize=(12, 6))
sns.barplot(x=district_crime.index, y=district_crime.values, palette='Blues_d')
plt.title('Percentage of Crimerate Distribution by District', fontsize=16)
plt.xlabel('District', fontsize=12)
plt.ylabel('Percentage of Total Crimes', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# In[160]:


# Illustration by Neighborhood


# In[162]:


#Top 10 Neighborhoods with Most Crimes
neighborhood_crime = crimerate['Neighborhood'].value_counts().head(10)


# In[164]:


# Plotting Neighborhood Crime Hotspots
plt.figure(figsize=(12, 6))
sns.barplot(x=neighborhood_crime.values, y=neighborhood_crime.index, palette='Reds_d')
plt.title('Top 10 Neighborhoods with Most Crimes', fontsize=16)
plt.xlabel('Crime Count', fontsize=12)
plt.ylabel('Neighborhood', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# In[166]:


# Visualizing Crimerate in Detail


# In[168]:


# Analyzing indoor vs outdoor crime: Understanding the distribution of crimes happening indoors and outdoors
IndoorOutdoor_crime_distribution = crimerate['Inside/Outside'].value_counts(normalize=True) * 100
print(IndoorOutdoor_crime_distribution)


# In[172]:


# Illustrating using a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x=IndoorOutdoor_crime_distribution.index, y=IndoorOutdoor_crime_distribution.values, palette='viridis')
plt.title('Indoor vs Outdoor Crime Distribution %', fontsize=16)
plt.xlabel('Crime Location (Indoor/Outdoor)', fontsize=12)
plt.ylabel('Percentage of Total Crimes', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# In[174]:


#Analyzing the frequency of weapon usage in the crimes


# In[176]:


#Count of each weapon type
weapon_usage = crimerate['Weapon'].value_counts()
print(weapon_usage)


# In[178]:


# Illustrating the weapon usage using a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=weapon_usage.index, y=weapon_usage.values, palette='coolwarm')
plt.title('Weapon Usage in Crimes', fontsize=16)
plt.xlabel('Weapon Type', fontsize=12)
plt.ylabel('Count of Crimes', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', alpha=0.3) 
plt.tight_layout()
plt.show()


# ## PREDICTIVE MODELLING

# In[181]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[183]:


# Prepare data for modeling
X = crimerate[['CrimeYear', 'CrimeMonth', 'CrimeDay', 'CrimeHours', 'District', 'Inside/Outside']]
X = pd.get_dummies(X, drop_first=True)
y = crimerate['CrimeCode']


# In[185]:


# Splitting data to test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[187]:


# Training using Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[191]:


# Predicting and evaluating model
y_pred = model.predict(X_test)


# In[193]:


print(classification_report(y_test, y_pred))


# In[195]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:




