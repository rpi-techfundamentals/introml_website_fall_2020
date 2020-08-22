#!/usr/bin/env python
# coding: utf-8

# # Jupyter Class Helper
# ---
# These files are used to configure and organize the website's contents.

# In[1]:


#%load_ext autoreload
#%autoreload 2
#%matplotlib inline


# In[2]:


# Always run this before any of the following cells
import pandas as pd
import numpy as np
import csv
import logging
import subprocess
import yaml
import builder as bd
from pathlib import Path
base_path=Path('..')
config_path = base_path / 'config'
cf=bd.load_yaml_file(config_path / 'config.yml')
excel_file= config_path / cf['excel_file']
class_path= base_path / cf['class']
content_path = class_path / 'content'


# In[3]:


# These load configuration from the excel files 
config = bd.load_yaml_file(class_path / '_config.yml') # Load the file.
toc = bd.load_yaml_file(class_path / '_toc.yml') # Load the file.
config_xl= pd.read_excel(excel_file, sheet_name = '_config_yml', header=None, index_col=None)
schedule= pd.read_excel(excel_file, sheet_name = 'Schedule',  index_col=None)
content={}
content['Before Class']= pd.read_excel(excel_file, sheet_name = 'Before',  index_col=None)
content['In Class']= pd.read_excel(excel_file, sheet_name = 'During',  index_col=None)
content['Assignment']= pd.read_excel(excel_file, sheet_name = 'Assignments',  index_col=None)


# In[4]:


#Create the syllabus link.
#The second value of the index postion of the syllabus on the before class content.
bd.create_syllabus(content['Before Class'],0,cf['syllabus_message'],content_path / 'syllabus.md', config['repository']['url'])


# In[5]:


#Fix in case individual tries to publish where session is NA. This isn't allowed. 
schedule.loc[schedule['Session'].isna(),'Publish']=0. 


# In[6]:


#Generate Links from the schedule to the sessions and within the other tables. 
schedule.loc[schedule['Publish']==1,'Location']=schedule.loc[schedule['Publish']==1,'Session'].apply(lambda x: '../sessions/session'+str(int(x)))
schedule.loc[schedule['Publish']==1,'Type']='Markdown'
schedule=bd.link_generator(schedule, 'Summary',config['repository']['url'],'Link')
content['Assignment']=bd.link_generator(content['Assignment'], 'Assignment',config['repository']['url'],'Starter')
content['Before Class']=bd.link_generator(content['Before Class'], 'Content',config['repository']['url'],'Link')
content['In Class']=bd.link_generator(content['In Class'], 'Content',config['repository']['url'],'Link')


# In[7]:


#Get the in class activities and prepare and output a markdown file. 
schedule_ic=schedule.merge(content['In Class'], left_on='Session', right_on='Session', how='left')
schedule_ic= schedule_ic.loc[schedule_ic['Content'].notnull(),['Week', 'Session', 'Date', 'Content']]
schedule_ic=bd.pandas_to_md(schedule_ic, content_path / 'in_class.md', 'In Class',         include = ['Week', 'Session', 'Date', 'Content'], header=cf['in_class_header'])


# In[8]:


#Get the before class activities and prepare and output a markdown file. 
schedule_bc=schedule.merge(content['Before Class'], left_on='Session', right_on='Session', how='left')
schedule_bc= schedule_bc.loc[schedule_bc['Content'].notnull(),['Week', 'Session', 'Date', 'Content']]
schedule_bc=bd.pandas_to_md(schedule_bc, content_path / 'preparation.md', 'Before Class',                              include = ['Week', 'Session', 'Date', 'Content'], header=cf['bc_class_header'])
schedule=schedule.merge(content['Assignment'], left_on='Session', right_on='Session', how='left')


# In[9]:


#Get the assignments and prepare and output a markdown file. 
assignments_new = schedule.loc[schedule['Assignment'].notnull(),['Week', 'Session', 'Date', 'Assignment', 'Due']]
assignments_new=bd.pandas_to_md(assignments_new, content_path / 'assignments.md', 'Assignments',                              include = ['Week', 'Session', 'Date', 'Assignment', 'Due'],header=cf['assignments_header'])


# In[10]:


#Output the schedule to markdown.
schedule=bd.pandas_to_md(schedule, content_path / 'schedule.md', 'Schedule',                              include = ['Week', 'Session', 'Date', 'Day', 'Topic', 'Summary', 'Assignment', 'Due'],header=cf['schedule_header'])


# In[11]:


#Generate Session Files
toc=bd.generate_sessions(config, toc, 2, schedule, class_path / 'sessions',content, ['Before Class', 'In Class', 'Assignment'])


# In[12]:


#Update the sessions to the yaml file.  Other updates to notebooks need to be done manually.
bd.update_yaml_file(class_path / '_toc.yml', toc)


# In[13]:


#TBD Make it so that notebooks will show up in _toc.yml. 

