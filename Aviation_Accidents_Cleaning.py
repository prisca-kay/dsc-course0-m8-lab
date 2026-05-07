#!/usr/bin/env python
# coding: utf-8

# # Aviation Accidents Analysis
# 
# You are part of a consulting firm that is tasked to do an analysis of commercial and passenger jet airline safety. The client (an airline/airplane insurer) is interested in knowing what types of aircraft (makes/models) exhibit low rates of total destruction and low likelihood of fatal or serious passenger injuries in the event of an accident. They are also interested in any general variables/conditions that might be at play. Your analysis will be based off of aviation accident data accumulated from the years 1948-2023. 
# 
# Our client is only interested in airplane makes/models that are professional builds and could potentially still be active. Assume a max lifetime of 40 years for a make/model retirement and make sure to filter your data accordingly (i.e. from 1983 onwards). They would also like separate recommendations for small aircraft vs. larger passenger models. **In addition, make sure that claims that you make are statistically robust and that you have enough samples when making comparisons between groups.**
# 
# 
# In this summative assessment you will demonstrate your ability to:
# - **Use Pandas to load, inspect, and clean the dataset appropriately.**
# - **Transform relevant columns to create measures that address the problem at hand.**
# - conduct EDA: visualization and statistical measures to systematically understand the structure of the data
# - recommend a set of airplanes and makes conforming to the client's request and identify at least *two* factors contributing to airplane safety. You must provide supporting evidence (visuals, summary statistics, tables) for each claim you make.

# ### Make relevant library imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Loading and Inspection

# ### Load in data from the relevant directory and inspect the dataframe.
# - inspect NaNs, datatypes, and summary statistics

# In[ ]:


df = pd.read_csv('data/AviationData.csv'un)
df.head()
    


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[8]:


df.shape


# In[11]:


df.describe()


# ## Data Cleaning

# ### Filtering aircrafts and events

# We want to filter the dataset to include aircraft that the client is interested in an analysis of:
# - inspect relevant columns
# - figure out any reasonable imputations
# - filter the dataset

# In[72]:


df.columns


# In[80]:


relevant_cols = [
    'Make', 'Model', 'Aircraft.damage',
    'Aircraft.Category', 'Country',
    'Total.Fatal.Injuries', 'Total.Serious.Injuries',
    'Weather.Condition', 'Broad.phase.of.flight'
]

df[relevant_cols].head()


# In[81]:


df[relevant_cols].info()
df[relevant_cols].isnull().sum()


# In[13]:


[col.strip() for col in df]


# In[19]:


# Inspect important columns
cols = ['Aircraft.Category', 'Purpose.of.flight', 'Amateur.Built', 'Number.of.Engines']
print(df[cols].head())

# Filter to airplane accidents
df = df[df['Aircraft.Category'].fillna('').str.contains('Airplane', case=False)]

# Keep non-amateur aircraft
df = df[df['Amateur.Built'] == 'No']

# Remove rows with missing engines
df = df[df['Number.of.Engines'].notna()]

print(f"Remaining rows: {df.shape[0]}")


# In[ ]:


df_fill= [col for col in df if col is NaN fillna('unkwnown')]


# ### Cleaning and constructing Key Measurables
# 
# Injuries and robustness to destruction are a key interest point for the client. Clean and impute relevant columns and then create derived fields that best quantifies what the client wishes to track. **Use commenting or markdown to explain any cleaning assumptions as well as any derived columns you create.**

# **Construct metric for fatal/serious injuries**
# 
# *Hint:* Estimate the total number of passengers on each flight. The likelihood of serious / fatal injury can be estimated as a fraction from this.

# In[20]:


# Fill missing injury counts
injury_cols = [
    'Total.Fatal.Injuries',
    'Total.Serious.Injuries',
    'Total.Minor.Injuries',
    'Total.Uninjured'
]

for col in injury_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Estimate total passengers
df['Estimated.Passengers'] = (
    df['Total.Fatal.Injuries'] +
    df['Total.Serious.Injuries'] +
    df['Total.Minor.Injuries'] +
    df['Total.Uninjured']
)

# Remove divide-by-zero rows
df = df[df['Estimated.Passengers'] > 0]

# Create injury metrics
df['Fatal.Serious.Count'] = (
    df['Total.Fatal.Injuries'] +
    df['Total.Serious.Injuries']
)

df['Fatal.Serious.Fraction'] = (
    df['Fatal.Serious.Count'] /
    df['Estimated.Passengers']
)

df[['Estimated.Passengers', 'Fatal.Serious.Fraction']].head()


# **Aircraft.Damage**
# - identify and execute any cleaning tasks
# - construct a derived column tracking whether an aircraft was destroyed or not.

# In[22]:


# Inspect aircraft damage
print(df['Aircraft.damage'].value_counts(dropna=False))

# Standardize labels
df['Aircraft.damage'] = (
    df['Aircraft.damage']
    .fillna('Unknown')
    .str.title()
)
# Create binary destruction variable
df['Destroyed'] = np.where(
    df['Aircraft.damage'] == 'Destroyed',
    1,
    0
)

df[['Aircraft.damage', 'Destroyed']].head()


# ### Investigate the *Make* column
# - Identify cleaning tasks here
# - List cleaning tasks clearly in markdown
# - Execute the cleaning tasks
# - For your analysis, keep Makes with a reasonable number (you can put the threshold at 50 though lower could work as well)

# In[23]:


print(df['Make'].value_counts().head(20))

# Standardize make labels
df['Make'] = (
    df['Make']
    .astype(str)
    .str.upper()
    .str.strip()
)

# Keep makes with at least 50 records
make_counts = df['Make'].value_counts()
valid_makes = make_counts[make_counts >= 50].index

df = df[df['Make'].isin(valid_makes)]

print(df['Make'].nunique())


# ### Inspect Model column
# - Get rid of any NaNs.
# - Inspect the column and counts for each model/make. Are model labels unique to each make?
# - If not, create a derived column that is a unique identifier for a given plane type.

# In[24]:


df['Model'] = df['Model'].fillna('UNKNOWN')

# Standardize model names
df['Model'] = (
    df['Model']
    .astype(str)
    .str.upper()
    .str.strip()
)

# Create unique plane type identifier
df['Plane.Type'] = df['Make'] + ' ' + df['Model']

print(df[['Make', 'Model', 'Plane.Type']].head())


# ### Cleaning other columns
# - there are other columns containing data that might be related to the outcome of an accident. We list a few here:
# - Engine.Type
# - Weather.Condition
# - Number.of.Engines
# - Purpose.of.flight
# - Broad.phase.of.flight
# 
# Inspect and identify potential cleaning tasks in each of the above columns. Execute those cleaning tasks. 
# 
# **Note**: You do not necessarily need to impute or drop NaNs here.

# In[25]:


cat_cols = [
    'Engine.Type',
    'Weather.Condition',
    'Purpose.of.flight',
    'Broad.phase.of.flight'
]

for col in cat_cols:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False).head())

# Standardize categories
for col in cat_cols:
    df[col] = (
        df[col]
        .fillna('Unknown')
        .astype(str)
        .str.title()
        .str.strip()
    )

# Numeric cleanup
df['Number.of.Engines'] = pd.to_numeric(
    df['Number.of.Engines'],
    errors='coerce'
)

df['Number.of.Engines'] = (
    df['Number.of.Engines']
    .fillna(df['Number.of.Engines'].median())
)

df.head()


# ### Column Removal
# - inspect the dataframe and drop any columns that have too many NaNs

# In[26]:


# Check missing value percentages
missing_pct = df.isna().mean().sort_values(ascending=False)

# Drop columns with more than 60% missing values
drop_cols = missing_pct[missing_pct > 0.60].index

print("Dropped columns:")
print(drop_cols)

df = df.drop(columns=drop_cols)

print(df.shape)


# ### Save DataFrame to csv
# - its generally useful to save data to file/server after its in a sufficiently cleaned or intermediate state
# - the data can then be loaded directly in another notebook for further analysis
# - this helps keep your notebooks and workflow readable, clean and modularized

# In[27]:


df.to_csv('cleaned_aviation_accidents.csv', index=False)

print("Cleaned dataset saved successfully.")

