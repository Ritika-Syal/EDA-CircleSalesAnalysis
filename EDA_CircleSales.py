#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[217]:


sales = pd.read_csv(r'F:\DUCAT\Data Science\k_circle_sales.csv')
sales.head()


# In[3]:


# Size of the dataset

sales.size


# In[215]:


# Rows and Columns of DataSet (row, columns)

sales.shape


# In[5]:


# Description about the data type or null values in the dataset

sales.info()


# In[6]:


sales.columns


# In[7]:


# Five Point Summary 

sales.describe().T


# # Data Analysis Types
# 
# 1. Descriptive : What happened? (Past events)
# 2. Diagnostic: Why did it happen? (Causes and reasons)
# 3. Predictive : What will happen? (Future outcomes)
# 4. Prescriptive : What should we do? (Decisions and actions)

# # Measures of Central Tendency
# They describe the center or typical value that best represents or summarizes the entire dataset — a central or common value around which most of the data points are clustered
# 
# 1. Mean : Average 
# 2. Median : Middle value (sorted data)
# 3. Mode : Most Frequent Value 

# # Measure of Dispersion
# 
# 1. Variance :  how far each number in a dataset is from the mean
# 2. Standard Deviation : how spread out the data is — but in the same units as the data
# 3. Range : Total spread of the dataset (max-min)
# 4. Quantiles or Quartiles : Divide data into equal-sized intervals or Divide data into 4 parts.

# In[92]:


# Selecting only numerical data type

num_cols = sales.select_dtypes(include = np.number).columns
num_cols


# In[9]:


sns.distplot(sales["Item_Weight"])
plt.show()


# In[10]:


# Univariate Analysis (numeric) - histogram , distribution 

plt.figure(figsize=(10,8))

a = 3
b = 2
c = 1
for i in num_cols:
    plt.subplot(a,b,c)
    sns.distplot(sales[i])
    c+=1
    plt.title(i)

plt.tight_layout()
plt.show()
    


# # Skewness 
# It is a measure of the asymmetry of the distribution of data.
# 
# It is of three types:
# 1. Symmetrical (No Skew)
# 2. Positive Skew (Right-Skewed)
# 3. Negative Skew (Left-Skewed)

# # 1. Symmetrical : Mean ≈ Median ≈ Mode
# * The data is evenly distributed around the center.
# * A perfect bell shaped curvev! (Balanced on both sides)
# 
# # 2. Positive Skew : Mean > Median > Mode
# * Tail on the right side is longer.
# * Data has a few very large values pulling the mean up.
# * Most data is on the left, with a long tail to the right.
# 
# # 3. Negative Skew: Mean < Median < Mode
# * Tail on the left side is longer.
# * Data has a few very small values pulling the mean down.
# * Most data is on the right, with a long tail to the left.

# | **Skewness Range**                      | **Interpretation**          | **Skew Type**             |
# | --------------------------------------- | --------------------------- | ------------------------- |
# | **-0.5 to +0.5**                        | **Approximately symmetric** | Not significantly skewed  |
# | **-1 to -0.5** or **0.5 to 1**          | **Moderately skewed**       | Slight left or right skew |
# | **Less than -1** or **greater than +1** | **Highly skewed**           | Strong left or right skew |
# 

# In[11]:


sales[num_cols].skew()


# #  Kurtosis 
# Defines the shape and according to shape it defines the data.
# How heavy or light the tails are compared to a normal distribution.
# 
# It is of three types:
# 1. Leptokurtic (High Kurtosis)
# 2. Mesokurtic (Normal)
# 3. Platykurtic (Low Kurtosis)

# # 1. Leptokurtic (Thin)
# * Kurtosis > 3 (or Excess > 0)
# * Tails are heavy → more outliers.
# * Sharp peak.
# 
# # 2. Mesokurtic 
# * Kurtosis ≈ 3 (or 0 if excess kurtosis is used)
# * Tails and peak are similar to the normal distribution.
# 
# # 3. Platykurtic (Thick/flat)
# * Kurtosis < 3 (or Excess < 0)
# * Tails are light → fewer outliers.
# * Flat peak.
# 

# | **Kurtosis Type** | **Kurtosis Value** | **Tails** | **Outliers**     | **Shape**         |
# | ----------------- | ------------------ | --------- | ---------------- | ----------------- |
# | Mesokurtic        | ≈ 3 (Excess = 0)   | Normal    | Moderate         | Normal bell curve |
# | Leptokurtic       | > 3 (Excess > 0)   | Heavy     | More (high risk) | Tall & narrow     |
# | Platykurtic       | < 3 (Excess < 0)   | Light     | Fewer            | Flat & wide       |
# 

# In[12]:


sales[num_cols].kurt()


# In[93]:


# Selecting only categorical data type

cat_cols = sales.select_dtypes(include = "object").columns
cat_cols


# In[94]:


cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
       'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
cat_cols


# In[15]:


sales["Item_Identifier"].nunique()


# In[16]:


# Univariate Analysis (Categorical columns) - countplot

plt.figure(figsize=(12,14))

c = 1 
for i in cat_cols:
    plt.subplot(3,2,c)
    sns.countplot(x = sales[i], data=sales)
    c+=1
    plt.title(i)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
    


# ##### Findings
# * Plotting numeric columns to check their shape, skewness and other errors using histogram (kde).
#     - We can see **Item Weight** have *Incorrect Values* (Yet to confirm).
#     - **Item Visibility, Item Outlet Sales** and **Profit** seemed to be *skewed* and *shape* is also *not normal*.
# * Plotting categorical columns to check the frequency and irregularities in categories using countplot.
#     - **Item Fat Content** and **Outlet Location Type** have *irregular Categories*.

# # Bivariate analysis (Predictor vs Target)
# Bivariate analysis is the process of exploring the relationship between two variables in a dataset — typically one predictor and one target.
# 
# 1. Num vs Num - scatterplot, lineplot, regplot
# 2. Cat vs Num - boxplot, violinplot, barplot, swarmplot
# 3. Cat vs Cat - countplot, frequency, stacked bar, heatmap

# In[17]:


sns.scatterplot(x = "Item_Weight", y = "Item_Outlet_Sales", data=sales)
plt.show()


# In[18]:


# Num vs Num 

plt.figure(figsize=(10,8))

c = 1 
for i in num_cols:
    plt.subplot(3,2,c)
    sns.scatterplot(x = sales[i], y = "Item_Outlet_Sales", data=sales)
    c+=1
    plt.title(i)

plt.tight_layout()
plt.show()
    


# In[19]:


# Cat vs Num - Boxplot (x= category,y=numeric, data=dataframe)

plt.figure(figsize=(16,20))

c = 1 
for i in cat_cols:
    plt.subplot(3,2,c)
    sns.boxplot(x = sales[i], y = "Item_Outlet_Sales", data=sales)
#     sns.barplot(x = sales[i], y = "Item_Outlet_Sales", data=sales)
#     sns.violinplot(x = sales[i], y = "Item_Outlet_Sales", data=sales)
#     sns.swarmplot(x = sales[i], y = "Item_Outlet_Sales", data=sales)
    c+=1
    plt.title(i)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
    


# In[20]:


sns.swarmplot(x='Item_Fat_Content',y='Item_Outlet_Sales',data=sales)
plt.show()


# In[21]:


# Trying different plots with different data item

plt.figure(figsize=(16, 20))

for c, i in enumerate(cat_cols):
   plt.subplot(3, 2, c+1)

   if c == 0:
       sns.barplot(x=sales[i], y="Item_Outlet_Sales", data=sales)
   elif c == 1:
       sns.boxplot(x=sales[i], y="Item_Outlet_Sales", data=sales)
   elif c == 2:
       sns.swarmplot(x=sales[i], y="Item_Outlet_Sales", data=sales)
   elif c == 3:
       sns.violinplot(x=sales[i], y="Item_Outlet_Sales", data=sales)
   elif c == 4:
       sns.violinplot(x=sales[i], y="Item_Outlet_Sales", data=sales)
   elif c == 5:
       sns.swarmplot(x=sales[i], y="Item_Outlet_Sales", data=sales)

   plt.title(i)
   plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# # Correlation
# It measures the linear relationship or strength and direction of the relationship between two numerical variables. It ranges from:
# 
# | **r Value** | **Interpretation**               |
# | ----------- | -------------------------------- |
# | `+1`        | Perfect **positive** correlation |
# | `0`         | **No** correlation               |
# | `-1`        | Perfect **negative** correlation |
# 
# 1. Positive Correlation : 
# As X increases, Y also increases
# 2. Negative Correlation : 
# As X increases, Y decreases
# 3. No Correlation :
# No clear relationship between X and Y

# In[22]:


# correlation 

sns.heatmap(sales[num_cols].corr(),annot=True)
plt.show()


# In[23]:


# Visualizimg Null Values

sns.heatmap(sales.isnull(), yticklabels=False, annot = False, cmap="YlGnBu")
plt.show()


# ##### Findings
# * Plotting both numerical and categorical columns against target variable to find Irregularities.
#     - Confirmed that **Item Weight** have some items with *Zero weight* and also detect some *outliers*.
#     - Categorical columns required some *feature engineering* and also deteched the outliers.
# * Also checked the skewness.
#     - **Item_Visibility** and **Item_Outlet_Sales** are *Rightly Skewed*.
#     - **Profit** is *Left Skewed*.
# * There is minimal coorelation between all the columns.
# 

# In[95]:


# Missing values 

sales.isnull().sum()


# In[96]:


# Percentage of Null Values

sales.isnull().mean()*100


# In[97]:


# loc : 
# loc[] is a label-based indexer in pandas. 
# It is used to access a group of rows and columns by labels or a boolean array.


# In[98]:


# Give all rows in the sales DataFrame where the Item_Weight is missing.

sales.loc[sales["Item_Weight"].isnull()]


# In[99]:


sales["Item_Weight"].mean()


# In[100]:


sales["Item_Weight"].median()


# In[101]:


# Grouping by Item_Identifier and selecting only Item_Weight column as we are working on that at present

sales.groupby("Item_Identifier")["Item_Weight"].mean()


# In[102]:


# Replacing null values with the mean of grouped columns to Item_weight 

val1 = sales.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
val1


# In[103]:


# Replacing the original column by appplying the treated data to actual dataset  
    
sales["Item_Weight"] = val1


# In[104]:


sales.head()


# In[105]:


sales[sales["Item_Weight"].isnull()]


# In[106]:


# Filling the left values


# In[107]:


val2 = sales.loc[(sales["Item_Fat_Content"] == "Low Fat") & (sales["Item_Type"] == "Snack Foods"), "Item_Weight"].mean()
val2


# In[108]:


val3 = sales.loc[(sales["Item_Fat_Content"] == "Regular") & (sales["Item_Type"] == "Baking Goods"), "Item_Weight"].mean()
val3


# In[109]:


# Filling the missing data values with mean 


# In[110]:


sales.loc[sales["Item_Identifier"] == "FDK57", "Item_Weight"] = np.round(val2,2)
sales.loc[sales["Item_Identifier"] == "FDQ60", "Item_Weight"] = np.round(val3,2)


# In[111]:


sales.loc[sales["Item_Weight"].isnull()]


# In[112]:


# checking

sales.isnull().sum()


# In[113]:


# For Category

sales['Outlet_Size'].mode()[0]


# In[114]:


sales['Outlet_Size'] = sales['Outlet_Size'].fillna(sales['Outlet_Size'].mode()[0])


# In[115]:


sales['Outlet_Location_Type'].mode()


# In[116]:



sales['Outlet_Location_Type'] = sales['Outlet_Location_Type'].fillna(sales['Outlet_Location_Type'].mode()[0])


# In[117]:


# Re-checking

sales.isnull().sum()


# ##### Findings
# * Using Heatmap to visually see which columns have null values andd also using percentage to see if they are treatable.
#     - Here, **Item Weight, Outlet_Size** and **Outlet_Location_Type** have null values less than 50% so we can treat them.
#     - Grouping **Item Weight** with **Item Identifiers** and applying *mean* and imputing the values in place of missing values.
#     - Rechecking and manually imputing values for the two values which got overlooked by *mean* based on their **Item Fat Content** and **Item Type**.
#     - For **Outlet Size** and **Outlet Location Type**, we are using *mode* to fill the null values.
# * Finally rechecking the whole data if any null value is present or not.

# # Feature Engineering
# It is the process of creating, transforming, or selecting features (columns) in a dataset to improve the performance of the machine learning model or gain better insights in EDA.
# 
# 1. Handling Missing Values: Fill missing values with mean or group mean
# 2. Encoding Categorical Variables: Label Encoding and One-Hot Encoding
# 3. Creating New Features: uncover hidden patterns, improve model performance, and bring domain knowledge into the dataset
# 4. Binning: Group into ranges as Low, Medium, High
# 5. Feature Transformation: Log-transform to reduce skewness
# 6. Date Feature Extraction (if you have date column): Extract year, month, day from a datetime column

# In[118]:


sales.head()


# In[119]:


sales["Item_Identifier"].nunique()


# In[120]:


# Creating new labels from existing one


# In[121]:


sales['Item_Identifier'][0][:2]


# In[122]:


items = []
for i in sales["Item_Identifier"]:
    items.append(i[:2])
#print(items)


# In[123]:


# Findig position of item_identifier
position = sales.columns.get_loc("Item_Identifier") + 1
position


# In[124]:


sales.insert(position, "Item_IDs", items) # Inserting at the specific position


# In[125]:


sales.rename(columns={"Item_IDs" : "Item_Codes"}, inplace=True)  # Renaming 


# In[126]:


sales.head()


# In[127]:


sales["Item_Fat_Content"].unique()


# In[128]:


# Replacing Unnecessary Values

sales.replace(to_replace=['Low Fat', 'Regular', 'low fat', 'LF', 'reg'],
              value=['Low Fat', 'Regular', 'Low Fat', 'Low Fat', 'Regular'], inplace=True)


# In[129]:


# Creating New Category for non edible items

sales.loc[sales["Item_Codes"] == "NC", "Item_Fat_Content"] = "Non-Consumable"


# In[130]:


sns.boxplot(x = "Item_Fat_Content", y="Item_Outlet_Sales", data=sales)
plt.show()


# In[131]:


sns.boxplot(x = "Item_Type", y = "Item_Outlet_Sales", data=sales)
plt.xticks(rotation=90)
plt.show()


# In[132]:


# For Item Type

sales["Item_Type"].unique()


# In[133]:


# Creating new labels from existing one

perishables = ['Dairy','Meat','Fruits and Vegetables','Breakfast','Breads','Starchy Foods', 'Seafood' ]
perishables


# In[134]:


def perish(x):
    if x in perishables:
        return "Perishables"
    else:
        return "Non-Perishables"


# In[135]:


# Using apply function to create new category

sales["Item_Type"] = sales["Item_Type"].apply(perish)


# In[136]:


sns.boxplot(x = "Item_Type", y = "Item_Outlet_Sales", data = sales) 
plt.show()


# In[137]:


# For Outlet Identifier

sns.boxplot(x="Outlet_Identifier", y = "Item_Outlet_Sales", data=sales)
plt.xticks(rotation = 90)
plt.show()


# In[138]:


sales.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean()  # analysis


# In[139]:


x = (sales.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean()) > 2400  # analysis
x[x]                                                                         # Returns true value only 


# In[140]:


x = (sales.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean()) < 1200  # analysis
x[x]                                                                         # Returns true value only 


# In[141]:


# Creating new labels from existing one

High = ["OUT027", "OUT035"]
Low = ["OUT010", "OUT019"] 


# In[142]:


def out_id(x):
    if x in High:
        return "High"
    if x in Low:
        return "Low"
    else:
        return "Medium"


# In[143]:


sales["Outlet_Identifier"] = sales["Outlet_Identifier"].apply(out_id)


# In[144]:


sns.boxplot(x="Outlet_Identifier", y = "Item_Outlet_Sales", data=sales)
plt.show()


# In[145]:


sns.boxplot(x = "Outlet_Establishment_Year", y = "Item_Outlet_Sales", data=sales)
plt.xticks(rotation=90)
plt.show()


# In[146]:


sales["Outlet_Size"].unique()


# In[147]:


sns.boxplot(x = "Outlet_Size", y = "Item_Outlet_Sales", data=sales)
plt.xticks(rotation=90)
plt.show()


# In[148]:


sales["Outlet_Location_Type"].unique()


# In[149]:


# Replacing Unnecessary Values

sales["Outlet_Location_Type"].replace(to_replace= ['nan', '  --', 'na', '  -', '?', 'NAN' ],
                                      value=["Missing"]*6, inplace=True)


# In[150]:


sns.boxplot( x = "Outlet_Location_Type", y = "Item_Outlet_Sales",data= sales) 
plt.show()


# In[151]:


sns.boxplot( x = "Outlet_Type", y = "Item_Outlet_Sales",data= sales) 
plt.xticks(rotation = 90)
plt.show()


# In[152]:


sns.scatterplot(x = "Item_Weight", y = "Item_Outlet_Sales", data=sales)
plt.show()


# In[153]:


# Correcting incorrect values
weights = np.round(sales.groupby("Item_Identifier")["Item_Weight"].mean(),2).to_dict()
weights


# In[154]:


sales["Item_Weight"] = sales["Item_Identifier"].map(weights)   # Mapping correct values


# In[155]:


sns.scatterplot(x = "Item_Weight", y = "Item_Outlet_Sales", data=sales)
plt.show()


# In[156]:


# Rechecking for incorrect values

sales.loc[sales["Item_Weight"] == 0]  


# In[157]:


# Imputing values

val4 = sales.loc[(sales["Item_Fat_Content"] == "Regular") & (sales["Item_Identifier"] == "FDN52"), "Item_Weight"].mean()
val4
sales.loc[(sales["Item_Identifier"] == "FDN52"), "Item_Weight"] = np.round(val4,2)


# In[158]:


# Dropping the other one ( we can also impute the value )

sales = sales.drop(sales[sales["Item_Identifier"] == "FDE52"].index)


# In[159]:


sns.scatterplot(x = "Item_Weight", y = "Item_Outlet_Sales", data=sales)
plt.show()


# In[160]:


sales.isnull().sum()


# ##### Findings
# * Creating New Labels .
#     - From Item Identifiers, extracting first characters and putting them in **Item Codes** to categorize items into broader groups.
#     - Correcting **Item Fat Content labels** as *Low Fat* and *Regular* as they contact similar labels with different names.
#     - For non edible items, we used **Item Code - NC** to put them in different label known as **Non - consumable** inside **Item Fat Content**.
#     - For **Item Type**, created a function to segregate all the items under two labels, *Perishables* and *Non - Perishables* from a predefined list of all **Item Type**.
#     - For **Item Outlet Sales**, created a function to segregate all the *outlets* under three labels, *Low*, *Medium* and *High* according to *sales*.
#     - For **Outlet Location Type**, we found the irregular values like *nan,?, etc.* under a single label named *Missing*.
#     - Didn't find anything irregular in **Outlet Type**.
# * Correcting incorrect values
#     - As discovered earlier, **Item Weight** contains *items* having *weights = zero*.
#     - To correct incorrect values, we are grouping the **Item Weights** with **Item Identifiers** and saving their *mean* inside a *dictationary* and then *mapping* the values in **Item Weight** Column.
#     - Checking for remaining incorrect values and manually imputing the values there and can also drop them.

# # Outliers 
# These are the data points that are significantly different from the rest of the data.
# They lie far away from the central values and can distort:
# 1. Averages (mean) : Can pull the average too high or low
# 2. Standard deviations : Boxplots and histograms look off
# 3. Machine learning models : Affects regression, clustering, etc.
# 
# # How to Detect Outliers
# 1. IQR Method (Interquartile Range)
# 2. Z-score Method
# 3. Percentile-based Capping

# | Strategy                    | Use Case                                       |
# | --------------------------- | ---------------------------------------------- |
# | **Remove**                  | If they are errors or rare cases               |
# | **Cap (Winsorize)**         | If you want to limit their influence           |
# | **Transform**               | Use `log`, `sqrt`, etc. to reduce skew         |
# | **Model-specific handling** | Some models (like tree-based) handle them well |
# 

# 1. IQR Method (Interquartile Range): Measures the spread of the middle 50% of the data.
# * Based on quartiles:
# * Q1 (25th percentile)
# * Q3 (75th percentile)
# * IQR = Q3 - Q1
# 
# 2. Z-Score Method (Standard Score): Measures how many standard deviations a value is from the mean.
# * Based on normal distribution.
# * Any data point with a Z-score > 3 or < -3 is considered an outlier.
# 
# 3. Percentile Method (Winsorization): Capping data at low and high percentiles.
# * Doesn’t remove outliers, but replaces them with percentile limits.
# * Cap values below the 10th percentile and above the 95th percentile.

# | Method         | Based On                 | Detects       | Common Thresholds         | Action          |
# | -------------- | ------------------------ | ------------- | ------------------------- | --------------- |
# | **IQR**        | Quartiles (Q1, Q3)       | Range-based   | 1.5 × IQR                 | Remove or Cap   |
# | **Z-Score**    | Mean & Std Dev           | Normal data   | Z > 3 or Z < -3           | Remove          |
# | **Percentile** | Distribution Percentiles | Mild outliers | 5th–95th, 10th–90th, etc. | Cap (Winsorize) |
# 

# In[161]:


# Outliers treatment 

plt.figure(figsize=(10,8))

c = 1 
for i in num_cols:
    plt.subplot(3,2,c)
    sns.boxplot(x = sales[i], data=sales)
    c+=1
    plt.title(i)
    #plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# In[162]:


# Capping Outliers with percentiles    

ll,ul = np.percentile(sales["Item_Visibility"],[10,95])
sales.loc[sales["Item_Visibility"]>ul, "Item_Visibility"]=ul
sales.loc[sales["Item_Visibility"]<ll,"Item_Visibility"]=ll


# In[163]:


# Visualising treated outliers 

plt.figure(figsize=(5,3))
sns.boxplot(x = sales['Item_Visibility'], data=sales)
plt.title("Item_Visibility")
plt.show()


# In[ ]:


# Outlier Trement for Profit


'''
from scipy.stats.mstats import winsorize

sales["Profit"] = winsorize(sales["Profit"], limits=[0.10, 0.10])  # Caps lowest & highest 10%
'''


# using winsorize but can also use quantile one

'''

lb, ub  = sales["Profit"].quantile([0.05,0.95])
sales["Profit_2"] = sales["Profit"].clip(lower = lb, upper = ub)

or

for i in num_cols:
    #q1,q3 = np.quantile(data[i],[0.25,0.75])
    #iqr = q3-q1
    #ul = q3+(1.5*iqr)    
    #ll = q1-(1.5*iqr)  
    ll,ul = np.quantile(sales[i],[0.10,0.95])
    sales.loc[sales[i]>ul,i]=ul
    sales.loc[sales[i]<ll,i]=ll

'''

# commenting just to prevent re application


# # Yeo-Johnson Transformation
# It is a power transformation technique used to make data more normally distributed (reduces skewness) which is useful for improving performance of machine learning models.
# It is similar to Box-Cox, but with one key advantage:
# 🟢 Yeo-Johnson works with both positive and negative values,
# 🔴 while Box-Cox only works with positive values.
# 
# 📌 Why Use It?
# * Handle skewed data
# * Improve normality (important for linear models, regression, etc.)
# * Reduce the impact of outliers
# * Stabilize variance
# 
# 🧠 When to Use Yeo-Johnson?
# * If data is not normally distributed
# * If features contain negative numbers.
# * Before applying models that assume normality (like linear regression, logistic regression, etc.)

# In[ ]:


# Importing Yeo - Johnson

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')


# In[ ]:


# Outlier Treatment for sales 

sales["Item_Outlet_Sales"] = pt.fit_transform(sales[["Item_Outlet_Sales"]])  


# | λ Value | Transformation Effect                          |
# | ------- | ---------------------------------------------- |
# | λ = 1   | No change (identity transformation)            |
# | λ = 0   | Log transformation                             |
# | λ < 1   | Compresses large values more (reduces skew)    |
# | λ > 1   | Expands values (useful if data is left-skewed) |
# | λ = 2   | Special case for Yeo-Johnson (negative log)    |

# In[164]:


# Checking for remaining one

q1 = sales["Item_Outlet_Sales"].quantile(0.25)
q3 = sales["Item_Outlet_Sales"].quantile(0.75)
iqr = q3 - q1

outliers = sales[(sales["Item_Outlet_Sales"] < q1 - 1.5*iqr) | (sales["Item_Outlet_Sales"] > q3 + 1.5*iqr)]
outliers


# In[165]:


# Removing Them

sales = sales[~sales.index.isin(outliers.index)]


# In[166]:


# Rechecking

plt.figure(figsize=(10,8))
c=1

for i in num_cols:
    plt.subplot(3,2,c)
    sns.boxplot(x = sales[i], data = sales)
    c+=1
    plt.title(i)
plt.tight_layout()
plt.show()


# In[167]:


sns.boxplot(x="Outlet_Identifier", y = "Item_Outlet_Sales", data=sales)
plt.show()


# In[169]:


# Treating for Outlet Identifier

s = sales.groupby("Outlet_Identifier")["Item_Outlet_Sales"]

qq1 = s.transform(lambda x : x.quantile(0.25))
qq3 = s.transform(lambda x : x.quantile(0.75))
iqrr = qq3 - qq1

outlierss = (sales["Item_Outlet_Sales"] < qq1 - 1.5 * iqrr) | (sales["Item_Outlet_Sales"] > qq3 + 1.5 * iqrr)
sales[outlierss].head()


# In[172]:


median_s = sales.groupby("Outlet_Identifier")["Item_Outlet_Sales"].transform("median")
# sales.loc[outlierss, "Item_Outlet_Sales"] = median_s
median_s 

# we can treat outliers here but decided not to as it is generally not necessary to treat outliers in category


# ##### Findings
# * Plotting boxplots to identify Outliers in numerical columns with target variable.
#     - **Item Visibility, Profit** and **Item Outlet Sales** have *Outliers*.
#     - Capping the *Outliers* in **Item Visibility** using *10th and 95th Percentile*.
#     - Using *Winsorize method* for managing *Outliers* in **Profit** by using the minimum value possible (lowest 10% and highest 10%).
#     - Applying *Yeo - Johnson Method* to reduce the effect of *Outliers* in **Item Outlet Sales** as we can't change the values by capping as the data is of *sales*.
#     - Detecting the remaining *Outliers* in **Item Outlet Sales** and it comes out to be four only. So we are dropping them.
# * Plotting the Boxplots for categorical columns
#     - Detected the *Outliers* in **Outlet Identifier**
#     - Grouping the **Outlet Identifier** with **Item Outlet Sales** and identifying the potential *Outliers* using *IQR*.
#     - Generally, It is not necessary to treat *Outliers* present in *Categorical Columns* and here also we didn't treat them but we can if according to the requirements.

# ### Skewness Check

# In[173]:


sales.head()


# In[174]:


sales.drop(columns = ["Item_Identifier"], inplace= True)


# In[175]:


sales["Outlet_Establishment_Year"] = sales["Outlet_Establishment_Year"].astype(str)
sales.dtypes


# In[176]:


num_cols


# In[177]:


num_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales', 'Profit'] # Removing Year
num_cols


# In[178]:


sales[num_cols].skew()


# In[179]:


# Plotting to check normality of data

plt.figure(figsize=(10,8))

c = 1
for i in num_cols:
    plt.subplot(3,2,c)
    sns.histplot(x = sales[i], data=sales, kde= True)
    c += 1
    plt.title(i)
plt.tight_layout()
plt.show()


# In[180]:


cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Establishment_Year']


# In[194]:


# Plotting Categorical Columns

plt.figure(figsize=(14,10))

c = 1
for i in cat_cols:
    plt.subplot(3,3,c)
    sns.countplot(x = sales[i], data = sales)
    c += 1
    plt.title(i)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ##### Findings
# * Preparing Data.
#     - Dropping the unnecesary **Item Identifier** Column.
#     - Changing the *data type* of **Outlet Establishment Year** from *Numeric* to *Category*.
#     - Checking the *skewness* of data and found that no column is *skewed*.
#     - Plotting the *Numerical columns* again to check the *Normality of data*.
#     - Plotting the *Categorical columns* again to check the *frequency distribution* and other aspects like *labels etc.*

# # Extra Experimental Part

# ### Scaling

# In[195]:


from scipy.stats import stats


# In[196]:


zscore = pd.DataFrame(stats.zscore(sales[num_cols]))
zscore


# In[197]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# In[198]:


sc = StandardScaler()
mm = MinMaxScaler()
rs = RobustScaler()


# In[199]:


scaled_sc = pd.DataFrame(sc.fit_transform(sales[num_cols]), columns=num_cols)
scaled_sc


# In[200]:


scaled_mm = pd.DataFrame(mm.fit_transform(sales[num_cols]), columns=num_cols)
scaled_mm


# In[201]:


scaled_rs = pd.DataFrame(rs.fit_transform(sales[num_cols]), columns=num_cols)
scaled_rs


# ### Encoding
# 

# In[203]:


# dummy
pd.get_dummies(sales[cat_cols])


# In[204]:


# One Hot (n-1)
pd.get_dummies(sales[cat_cols], drop_first = True)


# ### Splitting the data
# 

# In[205]:


from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression   can also be used 


# In[206]:


x = sales.drop("Item_Outlet_Sales", axis=1)
y = sales["Item_Outlet_Sales"]


# In[207]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.20, random_state=0)


# In[208]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ### Findings
# * Performed the ML methods like :
#     - Scaling using methods like *Zscore*, *Standard Scalar*, *MinMax Scaler* and *Robust Scalar*.
#     - Encoded the data using methods like *Dummy Encoding*, *One - Hot Encoding*.
#     - Splitting the data into *Train* and *Test* parts.
