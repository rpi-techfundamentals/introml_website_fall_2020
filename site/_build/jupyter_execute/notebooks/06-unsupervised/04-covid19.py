# Coronavirus Data Modeling


### Background
From Wikipedia...

"The 2019–20 coronavirus pandemic is an ongoing global pandemic of coronavirus disease 2019 (COVID-19) caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The virus was first reported in Wuhan, Hubei, China, in December 2019.[5][6] On March 11, 2020, the World Health Organization declared the outbreak a pandemic.[7] As of March 12, 2020, over 134,000 cases have been confirmed in more than 120 countries and territories, with major outbreaks in mainland China, Italy, South Korea, and Iran.[3] Around 5,000 people, with about 3200 from China, have died from the disease. More than 69,000 have recovered.[4]

The virus spreads between people in a way similar to influenza, via respiratory droplets from coughing.[8][9][10] The time between exposure and symptom onset is typically five days, but may range from two to fourteen days.[10][11] Symptoms are most often fever, cough, and shortness of breath.[10][11] Complications may include pneumonia and acute respiratory distress syndrome. There is currently no vaccine or specific antiviral treatment, but research is ongoing. Efforts are aimed at managing symptoms and supportive therapy. Recommended preventive measures include handwashing, maintaining distance from other people (particularly those who are sick), and monitoring and self-isolation for fourteen days for people who suspect they are infected.[9][10][12]

Public health responses around the world have included travel restrictions, quarantines, curfews, event cancellations, and school closures. They have included the quarantine of all of Italy and the Chinese province of Hubei; various curfew measures in China and South Korea;[13][14][15] screening methods at airports and train stations;[16] and travel advisories regarding regions with community transmission.[17][18][19][20] Schools have closed nationwide in 22 countries or locally in 17 countries, affecting more than 370 million students.[21]"

https://en.wikipedia.org/wiki/2019–20_coronavirus_pandemic 

For ADDITIONAL BACKGROUND, see JHU's COVID-19 Resource Center:
https://coronavirus.jhu.edu/




#RPI IDEA 

Check out these resources that IDEA has put together. 

https://idea.rpi.edu/covid-19-resources

### The Assignment

Our lives have been seriously disrupted by the coronavirus pandemic, and there is every indication that this is going to be a global event which requires colloration in a global community to solve.  Studying the data provides an opportunity to connect the pandemic to the variety of themes from the class. 

A number of folks have already been examining this data. 
https://ourworldindata.org/coronavirus-source-data


1. Discussion.  What is the role of open data?  Why is it important in this case?

2. Read this. 
https://medium.com/@tomaspueyo/coronavirus-act-today-or-people-will-die-f4d3d9cd99ca


What is the role of bias in the data?  Identify 2 different ways that the data could be biased.  

#Load some data
import pandas as pd
df=pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_daily_reports/10-21-2020.csv')
df

### Preprocessing
We have to deal with missing values first.

First let's check the missing values for each column. 

df.isnull().sum() 

df.loc[df['Province_State'].isnull(),:]

df.loc[df['Province_State'].notnull(),:]

### Missing Values and data
3. How might we deal with missing values? How is the data structured such that aggregation might be relevant.  





#Note the country is then the index here. 
country=pd.pivot_table(df, values=['Confirmed',	'Deaths',	'Recovered'], index='Country_Region',  aggfunc='sum')

country

### Clustering 

Here is and example of the elbow method, which is used to understand the number of clusters. 

https://scikit-learn.org/stable/modules/clustering.html#k-means

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion.

By looking at the total inertia at different numbers of clusters, we can get an idea of the appropriate number of clusters.



#This indicates the 

from sklearn.cluster import KMeans
sum_sq = {}
for k in range(1,30):
    kmeans = KMeans(n_clusters = k).fit(country)
    # Inertia: Sum of distances of samples to their closest cluster center
    sum_sq[k] = kmeans.inertia_
  
  

#ineria at different levels of K
sum_sq

## The Elbow Method

Not a type of criteria like p<0.05, but the elbow method you look for where the change in the variance explained from adding more clusters drops extensively. 

# plot elbow graph
import matplotlib
from matplotlib import pyplot as plt
plt.plot(list(sum_sq.keys()),
         list(sum_sq.values()),
        linestyle = '-',
        marker = 'H',
        markersize = 2,
        markerfacecolor = 'red')

## Looks like we can justify 5 clusters. 

See how adding the 5th doesn't really impact the total variance as much?  It might be interesting to do the analysis both at 4 and 5 and try to interpret. 

kmeans = KMeans(n_clusters=5)
kmeans.fit(country)
country['y_kmeans'] = kmeans.predict(country)




## Looks like they are mostly 0s.  Let's merge our data back together so we could get a clearer picture. 


loc=pd.pivot_table(df, values=['Lat','Long_'], index='Country_Region',  aggfunc='mean')
#loc['cluster']=y_kmeans
loc



alldata=country.merge(loc, left_index=True, right_index=True)

#join in our dataframes

alldata.to_csv("alldata.csv")  
alldata

#Alldata
from google.colab import files
files.download("alldata.csv")

alldata.sort_values('cluster', inplace=True)

#How do we interpret our clusters? 

alldata[alldata.cluster!=0]

#Details
pd.set_option('display.max_rows', 500)  #this allows us to see all rows. 
alldata[alldata.cluster==0]