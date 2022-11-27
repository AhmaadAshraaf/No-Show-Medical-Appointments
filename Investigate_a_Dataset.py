#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset - [no show appointments]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > **Tip**: In this section of the report, provide a brief introduction to the dataset you've selected/downloaded for analysis. Read through the description available on the homepage-links present [here](https://docs.google.com/document/d/e/2PACX-1vTlVmknRRnfy_4eTrjw5hYGaiQim5ctr9naaRd4V9du2B5bxpd8FEH3KtDgp8qVekw7Cj1GLk1IXdZi/pub?embedded=True). List all column names in each table, and their significance. In case of multiple tables, describe the relationship between tables. 
# 
# 
# ### Question(s) for Analysis
# Questions we are trying to answer with our project :
# 
# 1- How many of the patients show to the scheduled appointment?
# 
# 2- What is the dominant gender that show up to the appointment?

# In[47]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(15, 8)}, font_scale=1.3)


# In[48]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# ### General Properties
# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.

# In[49]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
pd.options.display.max_rows = 9999
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')


# In[50]:


df


# In[ ]:





# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[51]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
df.sample(6)


# In[52]:


df.head()


# In[53]:


df.tail()


# In[54]:


df.info()


# In[55]:


df.describe()


# In[56]:


df.isnull().sum()


# In[57]:


df.nunique()


# In[58]:


for col in df.columns:
    print("unique values in "+ col+"column")
    print('-'*100)
    print(df[col].unique())
    print('-'*100)


# In[59]:


df.duplicated().sum()


# In[60]:


df.columns


# In[61]:


df.rename(columns={'ScheduledDay':'Scheduled_Date','AppointmentDay':'Appointment_Date', 'No-show':'No_show', 'Handcap':'Handicap'}, inplace=True)
df


# In[62]:


df['Scheduled_Date'] = pd.to_datetime(df['Scheduled_Date'])
df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'])
df.info()


# In[63]:


df['Scheduled_Month_Name'] = df['Scheduled_Date'].dt.month_name()
df['Scheduled_Day'] = df['Scheduled_Date'].dt.day
df['Scheduled_Day_Name'] = df['Scheduled_Date'].dt.day_name()
df['Scheduled_Hour'] = df['Scheduled_Date'].dt.hour


# In[64]:


df['Scheduled_Date'] = pd.to_datetime(df['Scheduled_Date'].dt.date)


# In[65]:


df['Appointment_Month_Name'] = df['Appointment_Date'].dt.month_name()
df['Appointment_Day'] = df['Appointment_Date'].dt.day
df['Appointment_Day_Name'] = df['Appointment_Date'].dt.day_name()
df['Appointment_Hour'] = df['Appointment_Date'].dt.hour


# In[66]:


df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'].dt.date)


# In[67]:


df['Waiting_Days'] = (df['Appointment_Date'] - df['Scheduled_Date']).dt.days


# In[68]:


df['Waiting_Days'].unique()


# In[69]:


np.sort(df['Waiting_Days'].unique())


# In[70]:


df['Waiting_Days'].value_counts().sort_index().head(5)


# In[71]:


df.drop(df[df['Waiting_Days']<0].index,inplace= True)


# In[72]:


#Handling handicap values
df['Handicap'].value_counts()


# In[73]:


df['Handicap'] = df['Handicap'].apply(lambda value : 1 if value>0 else 0)


# In[74]:


df['Handicap'].value_counts()


# In[75]:


#Handling values in age column
df['Age'].unique()


# In[76]:


np.sort(df['Age'].unique())


# In[77]:


df[df['Age']== -1]


# In[78]:


df.drop(df[df['Age']== -1].index, inplace= True)


# In[79]:


#Handling 0,1 values into yes and no values
dict_yes_no = {0 : 'no', 1 : 'yes'}
col_handled = ['Scholarship', 'Hipertension', 'Alcoholism','Handicap','Diabetes', 'SMS_received']
for col in col_handled:
    df[col] = df[col].map(dict_yes_no)


# In[80]:


#Printing unique values
for col in df.columns:
    print("Unique Values in "+ col+"column")
    print('-'*100)
    print(df[col].unique())
    print('-'*100)


# In[81]:


df


# In[82]:


df.drop(columns=['PatientId','AppointmentID'], axis= 1, inplace= True)


# In[83]:


df


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 
# 
# 
# 
# > **Tip**: - Investigate the stated question(s) from multiple angles. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables. You should explore at least three variables in relation to the primary question. This can be an exploratory relationship between three variables of interest, or looking at how two independent variables relate to a single dependent variable of interest. Lastly, you  should perform both single-variable (1d) and multiple-variable (2d) explorations.
# 
# 
# ### Research Question 1 (How many people show up or no show to appointments!)

# In[84]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
# How many people show-up and no-show to appointments
sns.set(rc={'figure.figsize':(10, 8)}, font_scale= 1.3)
plt.bar(['Show-up' , 'No-show'], df['No_show'].value_counts(),facecolor= 'm', edgecolor= 'k')
plt.title('Show-up or No-show appointments count', size= 20)
plt.show()


# In[85]:


plt.pie(df['No_show'].value_counts(),labels=['Show-up','No-show'],shadow= True, autopct= '%.2f%%',colors=['m','r'],explode=[0.1,0.1])
plt.title('Show-up or No-show appointments count', size= 20)
plt.axis('equal')
plt.show()


# In[86]:


# Conclusion; Most of patients showed up to the appointments


# ### Research Question 2  (Do males show up to appointments more than females?)

# In[87]:


# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.
showed = df[df['No_show']=='No']
un_showed = df[df['No_show']=='Yes']


# In[88]:


un_showed['Gender']


# In[89]:


sns.countplot(x=df['Gender'],hue=df['No_show'],palette='dark')
plt.title('Show-Up or No-Show vs Gender',size = 20)
plt.show()


# In[90]:


labels = ['Female', 'Male']
fig, ax = plt.subplots(figsize=(10,8))
ax.bar(labels, showed['Gender'].value_counts(), label='Showed_Up'
                       ,facecolor ='g',edgecolor='k')
ax.bar(labels, un_showed['Gender'].value_counts(), bottom=showed['Gender'].value_counts(),
       label="No_Show",facecolor ='r',edgecolor='k')
plt.title('Show-Up or No-Show vs Gender',size = 20)
ax.legend()


# <a id='conclusions'></a>
# ## Conclusions
# 
# 1- Most of patients show up to the scheduled appointments.
# 2- Females show to the appointments more than males.
# 
# ##LIMITATIONS: 
# missing information that would be useful in assesing the patients that show or no-show to appointments as:
# 1- medical insurance.
# 2- wether the patient is employed or not. 
# ## Submitting your Project 
# 
# > **Tip**: Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > **Tip**: Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > **Tip**: Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[91]:


#Conclusions:
# Most of patients showed up to the appointments
# Females showed up to the appointments more than males


# In[92]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




