#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis(EDA) on iris data set by python.

# In[1]:


#imported Python libraries numpy,pandas,matplotlib and seaborn with creating there standard alias np,pd,plt,sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#loading the iris dataset into pandas
df=pd.read_csv("C:\\Users\\AMAN BIRADAR\\Downloads\\Python\\iris.csv")
#head(5)displays 1st five rows for dataset
df.head(5)


# In[3]:


#prints no. of row and columns of iris dataset
print(df.shape)


# In[4]:


#displays name of columns of iris dataset
print(df.columns)


# In[5]:


#value_counts() is used to display the count 
#df is a balanced data set as all the 3 species(setosa,virginica,versicolor) have the no of data points is 50 for every class.
df['species'].value_counts()


# # 1-D Scatter Plot
# 

# In[45]:


#1D Scatter Plot
df_setso = df.loc[df['species'] == 'setosa'];
df_virginica = df.loc[df['species'] == 'virginica'];
df_versicolor = df.loc[df['species'] == 'versicolor'];
plt.plot(df_setso['petal_length'],np.zeros_like(df_setso['petal_length']), 'o')
plt.plot(df_versicolor['petal_length'],np.zeros_like(df_versicolor['petal_length']), 'o')
plt.plot(df_virginica['petal_length'],np.zeros_like(df_virginica['petal_length']), 'o')

plt.grid()
plt.show()


# #observations of 1-D Scatter plot
# 
# - Blue points are setosa,orange are versicolor,and green are virginica.
# - we can see a lot of overlapping is there for versicolor and virginica.
# - 1D Scatter are very hard to read and understand
# 

# # 2-D Scatter plot

# In[7]:


#2D Scatter plot
df.plot(kind="scatter",x="sepal_length",y="sepal_width")
plt.show()


# Observation:
# - Using matplotlib above 2D Scatter plot is plotted
# - We have plotted sepal_length on the x-axis and sepal_width on the y-axis. 
# - We are not able to understand which is setosa,versicolor and virginica for the above plot.
# - All points are in same colour so,cannot make much sense out it
# 
# 

# In[8]:


#2D Scatter Plot 
sns.set_style('whitegrid');
sns.FacetGrid(df,hue='species',height=5)     .map(plt.scatter,'sepal_length','sepal_width')     .add_legend()
plt.show()


# #Observation:
# - Blue points can be easily separated from orange and green by drawing a line.
# - But orange and green data points cannot be easily separated.
# - Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others.
# - Separating Versicolor from Viginica is much harder as they have considerable overlap.
# 

# # 3-D Scatter Plot

# In[9]:


#3D Scatter plot
plt.figure(figsize=(10,5))
from  mpl_toolkits import mplot3d
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(df['sepal_length'],df['sepal_width'],df['petal_length'],c='r',marker='o')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length')


# Observation:
# - Using matplotlib above 3D Scatter plot is plotted
# - 3D plot can classify more in detail than 2D plot
# - We have plotted sepal_length on the x-axis,sepal_width on the y-axis and petal_length on z-axis. 
# - We are not able to understand which is setosa,versicolor and virginica for the above plot.
# - All points are in same colour so,cannot make much sense out it
# 

# In[10]:


#3D Scatter Plot
import plotly.express as px
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',color='species')
fig.show()


# #Observation:
# - 3D plot will be used for three variables or dimensions.
# - Blue points can be easily separated from orange and green by drawing a line.
# - But orange and green data points cannot be easily separated.
# - Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others.
# - Separating Versicolor from Viginica is much harder as they have considerable overlap.
# 

# # What would do if we have more than 3 dimensions or features in our dataset as we humans do have the capability to visualize more than 3 dimensions?
#  One solution to this problem is pair plots.
#  
#  
# # Pair plots

#  A pairs plot allows us to see both distribution of single variables and relationships between two variables.
#  - sepal length, sepal width
#  - sepal length, petal length
#  - sepal length, petal width
#  - sepal width, petal length
#  - sepal width, petal width
#  - petal length, petal width
# 
# So, here instead of trying to visualize four dimensions which is not possible. We will look into 6 2D plots and try to understand the 4-dimensional data in the form of a matrix.
# 

# In[11]:


sns.pairplot(df,hue='species',height=3)


# The diagonal plot which showcases the histogram. The histogram allows us to see the PDF/Probability distribution of a single variable
# 
# Upper triangle and lower triangle which shows us the scatter plot.
# 
# The scatter plots show us the relationship between the features. These upper and lower triangles are the mirror image of each other.
# 
# Observation
# - petal length and petal width are the most useful features to identify various flower types.
# - While Setosa can be easily identified (linearly separable), virginica and Versicolor have some overlap (almost linearly separable).
# - We can find “lines” and “if-else” conditions to build a simple model to classify the flower types.

# # Histogram and Introduction of PDF

# In[39]:


sns.FacetGrid(df,hue='species',height=5)     .map(sns.distplot,'petal_length')     .add_legend();
    
plt.show();


# # Univariate Analysis using PDF

# In[37]:


sns.FacetGrid(df,hue='species',height=5)     .map(sns.distplot,'petal_width')     .add_legend();
    
plt.show();


# In above plot there is overlap between vericolor and virginca
# 

# In[38]:


sns.FacetGrid(df,hue='species',height=5)     .map(sns.distplot,'sepal_width')     .add_legend();
    
plt.show();


# In this above plot virginica and versicolor are fully overlapped.

# In[22]:


sns.FacetGrid(df,hue='species',height=5)     .map(sns.distplot,'sepal_length')     .add_legend();
    
plt.show();


# In this classes cannot be separated because all of them are overlapped

# # CDF(Cumulative distribution function)
# 

# In[41]:


df_setosa = df.loc[df['species'] == 'setosa'];
df_virginica = df.loc[df['species'] == 'virginica'];
df_versicolor = df.loc[df['species'] == 'versicolor'];
counts, bin_edges = np.histogram(df_setosa['petal_length'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf); 


# In[27]:


print(bin_edges);


# In[28]:


cdf = np.cumsum(pdf)
plt.grid()
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf);


# In above plot the blue line is pdf and the orange line is cdf.

# # Mean, Variance and Standard Deviation

# Mean is average of a given set of data.
# 
# Variance is the sum of squares of differences between all numbers and means.
# 
# Standard Deviation is square root of variance. It is a measure of the extent to which data varies from the mean.

# In[40]:




print('Means:')
print(np.mean(df_setosa['petal_length']))
print(np.mean(np.append(df_setosa['petal_length'],50)));
print(np.mean(df_virginica['petal_length']))
print(np.mean(df_versicolor['petal_length']))
print('\nStd-dev:');
print(np.std(df_setosa['petal_length']))
print(np.std(df_virginica['petal_length']))
print(np.std(df_versicolor['petal_length']))


# # Meadian:
# We can see that mean,variance ,std-dev can be easily corrupted by outliers.
# 
# Median:median is the middle value from the sorted values

# # Box plot with whisker

# Box plot is very useful in detecting whether there is any outliers in our dataset or whether a distribution is skewed or not.

# In[34]:


sns.boxplot(x='species',y='petal_length', data=df)
plt.show()


# # Violin plot

# A violin plot is a method of plotting numeric data.
# 
# It is similar to a box plot, with the addition of a rotated kernel density plot on each side
# 
# It shows the probability density of the data at different values, usually smoothed by a kernel density estimator.

# In[35]:


sns.violinplot(x='species',y='petal_length', data=df, height=8)
plt.show()


# # Conclusion:
# Performing EDA we are able to understand which features are important to apply machine learning algorithm.
