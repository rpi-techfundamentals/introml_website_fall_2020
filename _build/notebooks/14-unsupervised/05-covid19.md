---
interact_link: content/notebooks/14-unsupervised/05-covid19.ipynb
kernel_name: python3
has_widgets: false
title: 'COVID-19 Cluster'
prev_page:
  url: /notebooks/14-unsupervised/04-regression-feature-selection.html
  title: 'Feature Selection and Importance'
next_page:
  url: /notebooks/16-intro-nlp/01-titanic-features.html
  title: 'Titanic Feature Creation'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


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



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```###Answer here.
q1="""

"""

```
</div>

</div>



2. Read this. 
https://medium.com/@tomaspueyo/coronavirus-act-today-or-people-will-die-f4d3d9cd99ca


What is the role of bias in the data?  Identify 2 different ways that the data could be biased.  



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```###Answer here. 
q2="""

"""

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Load some data
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/03-22-2020.csv')
df

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Last Update</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hubei</td>
      <td>China</td>
      <td>2020-03-22T09:43:06</td>
      <td>67800</td>
      <td>3144</td>
      <td>59433</td>
      <td>30.9756</td>
      <td>112.2707</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Italy</td>
      <td>2020-03-22T18:13:20</td>
      <td>59138</td>
      <td>5476</td>
      <td>7024</td>
      <td>41.8719</td>
      <td>12.5674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Spain</td>
      <td>2020-03-22T23:13:18</td>
      <td>28768</td>
      <td>1772</td>
      <td>2575</td>
      <td>40.4637</td>
      <td>-3.7492</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Germany</td>
      <td>2020-03-22T23:43:02</td>
      <td>24873</td>
      <td>94</td>
      <td>266</td>
      <td>51.1657</td>
      <td>10.4515</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Iran</td>
      <td>2020-03-22T14:13:06</td>
      <td>21638</td>
      <td>1685</td>
      <td>7931</td>
      <td>32.4279</td>
      <td>53.6880</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>NaN</td>
      <td>Jersey</td>
      <td>2020-03-17T18:33:03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49.1900</td>
      <td>-2.1100</td>
    </tr>
    <tr>
      <th>305</th>
      <td>NaN</td>
      <td>Puerto Rico</td>
      <td>2020-03-22T22:43:02</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18.2000</td>
      <td>-66.5000</td>
    </tr>
    <tr>
      <th>306</th>
      <td>NaN</td>
      <td>Republic of the Congo</td>
      <td>2020-03-17T21:33:03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.4400</td>
      <td>15.5560</td>
    </tr>
    <tr>
      <th>307</th>
      <td>NaN</td>
      <td>The Bahamas</td>
      <td>2020-03-19T12:13:38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24.2500</td>
      <td>-76.0000</td>
    </tr>
    <tr>
      <th>308</th>
      <td>NaN</td>
      <td>The Gambia</td>
      <td>2020-03-18T14:13:56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.4667</td>
      <td>-16.6000</td>
    </tr>
  </tbody>
</table>
<p>309 rows × 8 columns</p>
</div>
</div>


</div>
</div>
</div>



### Preprocessing
We have to deal with missing values first.

First let's check the missing values for each column. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```df.isnull().sum() 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Province/State    174
Country/Region      0
Last Update         0
Confirmed           0
Deaths              0
Recovered           0
Latitude            0
Longitude           0
dtype: int64
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```df.loc[df['Province/State'].isnull(),:]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Last Update</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Italy</td>
      <td>2020-03-22T18:13:20</td>
      <td>59138</td>
      <td>5476</td>
      <td>7024</td>
      <td>41.8719</td>
      <td>12.5674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Spain</td>
      <td>2020-03-22T23:13:18</td>
      <td>28768</td>
      <td>1772</td>
      <td>2575</td>
      <td>40.4637</td>
      <td>-3.7492</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Germany</td>
      <td>2020-03-22T23:43:02</td>
      <td>24873</td>
      <td>94</td>
      <td>266</td>
      <td>51.1657</td>
      <td>10.4515</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Iran</td>
      <td>2020-03-22T14:13:06</td>
      <td>21638</td>
      <td>1685</td>
      <td>7931</td>
      <td>32.4279</td>
      <td>53.6880</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Korea, South</td>
      <td>2020-03-22T11:13:17</td>
      <td>8897</td>
      <td>104</td>
      <td>2909</td>
      <td>35.9078</td>
      <td>127.7669</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>304</th>
      <td>NaN</td>
      <td>Jersey</td>
      <td>2020-03-17T18:33:03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49.1900</td>
      <td>-2.1100</td>
    </tr>
    <tr>
      <th>305</th>
      <td>NaN</td>
      <td>Puerto Rico</td>
      <td>2020-03-22T22:43:02</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18.2000</td>
      <td>-66.5000</td>
    </tr>
    <tr>
      <th>306</th>
      <td>NaN</td>
      <td>Republic of the Congo</td>
      <td>2020-03-17T21:33:03</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.4400</td>
      <td>15.5560</td>
    </tr>
    <tr>
      <th>307</th>
      <td>NaN</td>
      <td>The Bahamas</td>
      <td>2020-03-19T12:13:38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24.2500</td>
      <td>-76.0000</td>
    </tr>
    <tr>
      <th>308</th>
      <td>NaN</td>
      <td>The Gambia</td>
      <td>2020-03-18T14:13:56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.4667</td>
      <td>-16.6000</td>
    </tr>
  </tbody>
</table>
<p>174 rows × 8 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```df.loc[df['Province/State'].notnull(),:]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Last Update</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hubei</td>
      <td>China</td>
      <td>2020-03-22T09:43:06</td>
      <td>67800</td>
      <td>3144</td>
      <td>59433</td>
      <td>30.9756</td>
      <td>112.2707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>France</td>
      <td>2020-03-22T23:43:02</td>
      <td>16018</td>
      <td>674</td>
      <td>2200</td>
      <td>46.2276</td>
      <td>2.2137</td>
    </tr>
    <tr>
      <th>6</th>
      <td>New York</td>
      <td>US</td>
      <td>2020-03-22T22:13:32</td>
      <td>15793</td>
      <td>117</td>
      <td>0</td>
      <td>42.1657</td>
      <td>-74.9481</td>
    </tr>
    <tr>
      <th>9</th>
      <td>United Kingdom</td>
      <td>United Kingdom</td>
      <td>2020-03-22T22:43:03</td>
      <td>5683</td>
      <td>281</td>
      <td>65</td>
      <td>55.3781</td>
      <td>-3.4360</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Netherlands</td>
      <td>Netherlands</td>
      <td>2020-03-22T14:13:10</td>
      <td>4204</td>
      <td>179</td>
      <td>2</td>
      <td>52.1326</td>
      <td>5.2913</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>294</th>
      <td>From Diamond Princess</td>
      <td>Australia</td>
      <td>2020-03-14T02:33:04</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>35.4437</td>
      <td>139.6380</td>
    </tr>
    <tr>
      <th>297</th>
      <td>French Guiana</td>
      <td>France</td>
      <td>2020-03-18T14:33:15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4.0000</td>
      <td>-53.0000</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Guadeloupe</td>
      <td>France</td>
      <td>2020-03-18T14:33:15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16.2500</td>
      <td>-61.5833</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Mayotte</td>
      <td>France</td>
      <td>2020-03-18T14:33:15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-12.8431</td>
      <td>45.1383</td>
    </tr>
    <tr>
      <th>300</th>
      <td>Reunion</td>
      <td>France</td>
      <td>2020-03-18T14:33:15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-21.1351</td>
      <td>55.2471</td>
    </tr>
  </tbody>
</table>
<p>135 rows × 8 columns</p>
</div>
</div>


</div>
</div>
</div>



### Data Reporting
#TBD

For the last update value, we could create a feature that as equal to the number of days since the last report. We might eliminate data that is too old. 





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#TBD For the last update value, we could create a feature that as equal to the number of 

```
</div>

</div>



### Missing Values and data
3. How might we deal with missing values? How is the data structured such that aggregation might be relevant.  





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```###Answer here. 
q3="""

"""

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Note the country is then the index here. 
country=pd.pivot_table(df, values=['Confirmed',	'Deaths',	'Recovered'], index='Country/Region',  aggfunc='sum')

```
</div>

</div>



### Clustering

Here is and example of the elbow method, which is used to understand the number of clusters. 

https://scikit-learn.org/stable/modules/clustering.html#k-means

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion.

By looking at the total inertia at different numbers of clusters, we can get an idea of the appropriate number of clusters.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This indicates the 

from sklearn.cluster import KMeans
sum_sq = {}
for k in range(1,30):
    kmeans = KMeans(n_clusters = k).fit(country)
    # Inertia: Sum of distances of samples to their closest cluster center
    sum_sq[k] = kmeans.inertia_
  
  

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#ineria at different levels of K
sum_sq

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
{1: 18388732383.6612,
 2: 5781628112.668509,
 3: 1437012534.1559324,
 4: 437453272.5568181,
 5: 249173713.07080925,
 6: 133720143.75294116,
 7: 101140484.75294116,
 8: 70970124.11542442,
 9: 41676542.30886076,
 10: 30017447.30886076,
 11: 16760158.796828683,
 12: 8796949.343951093,
 13: 4947365.970771144,
 14: 3797381.970771144,
 15: 2651536.5041044774,
 16: 1877813.419029374,
 17: 1363350.3819298667,
 18: 1024544.2380769933,
 19: 722546.6442815806,
 20: 643241.4546326294,
 21: 509133.6414066776,
 22: 418223.49140667764,
 23: 317512.6517577265,
 24: 261568.8043432082,
 25: 216271.43865631748,
 26: 184034.2586798829,
 27: 153780.61576464615,
 28: 127928.16630164167,
 29: 109987.48092580127}
```


</div>
</div>
</div>



## The Elbow Method

Not a type of criteria like p<0.05, but the elbow method you look for where the change in the variance explained from adding more clusters drops extensively. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# plot elbow graph
import matplotlib
from matplotlib import pyplot as plt
plt.plot(list(sum_sq.keys()),
         list(sum_sq.values()),
        linestyle = '-',
        marker = 'H',
        markersize = 2,
        markerfacecolor = 'red')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[<matplotlib.lines.Line2D at 0x7fdd6261ae48>]
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/notebooks/14-unsupervised/05-covid19_21_1.png)

</div>
</div>
</div>



#Looks like we can justify 4 clusters.

See how adding the 5th doesn't really impact the total variance as much?  It might be interesting to do the analysis both at 4 and 5 and try to interpret. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```kmeans = KMeans(n_clusters=4)
kmeans.fit(country)
y_kmeans = kmeans.predict(country)


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```y_kmeans

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0], dtype=int32)
```


</div>
</div>
</div>



## Looks like they are mostly 0s.  Let's merge our data back together so we could get a clearer picture.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```loc=pd.pivot_table(df, values=['Latitude',	'Longitude'], index='Country/Region',  aggfunc='mean')
loc['cluster']=y_kmeans
loc

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>cluster</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>33.9391</td>
      <td>67.7100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>41.1533</td>
      <td>20.1683</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>28.0339</td>
      <td>1.6596</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>42.5063</td>
      <td>1.5218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>-11.2027</td>
      <td>17.8739</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Uzbekistan</th>
      <td>41.3775</td>
      <td>64.5853</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Venezuela</th>
      <td>6.4238</td>
      <td>-66.5897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>14.0583</td>
      <td>108.2772</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>-13.1339</td>
      <td>27.8493</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>-19.0154</td>
      <td>29.1549</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 3 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#join in our dataframes
alldata= country.join(loc)
alldata.to_csv("alldata.csv")  
alldata

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>cluster</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>40</td>
      <td>1</td>
      <td>1</td>
      <td>33.9391</td>
      <td>67.7100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>89</td>
      <td>2</td>
      <td>2</td>
      <td>41.1533</td>
      <td>20.1683</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>201</td>
      <td>17</td>
      <td>65</td>
      <td>28.0339</td>
      <td>1.6596</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>113</td>
      <td>1</td>
      <td>1</td>
      <td>42.5063</td>
      <td>1.5218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>-11.2027</td>
      <td>17.8739</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Uzbekistan</th>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>41.3775</td>
      <td>64.5853</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Venezuela</th>
      <td>70</td>
      <td>0</td>
      <td>15</td>
      <td>6.4238</td>
      <td>-66.5897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>113</td>
      <td>0</td>
      <td>17</td>
      <td>14.0583</td>
      <td>108.2772</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-13.1339</td>
      <td>27.8493</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-19.0154</td>
      <td>29.1549</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 6 columns</p>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Alldata
from google.colab import files
files.download("alldata.csv")

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```alldata.sort_values('cluster', inplace=True)

```
</div>

</div>



#How do we interpret our clusters?



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```alldata[alldata.cluster!=0]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>cluster</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>81397</td>
      <td>3265</td>
      <td>72362</td>
      <td>32.729748</td>
      <td>111.684242</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>24873</td>
      <td>94</td>
      <td>266</td>
      <td>51.165700</td>
      <td>10.451500</td>
      <td>2</td>
    </tr>
    <tr>
      <th>US</th>
      <td>33276</td>
      <td>417</td>
      <td>178</td>
      <td>38.112296</td>
      <td>-84.664082</td>
      <td>2</td>
    </tr>
    <tr>
      <th>France</th>
      <td>16044</td>
      <td>674</td>
      <td>2200</td>
      <td>3.320689</td>
      <td>-13.517378</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Iran</th>
      <td>21638</td>
      <td>1685</td>
      <td>7931</td>
      <td>32.427900</td>
      <td>53.688000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>28768</td>
      <td>1772</td>
      <td>2575</td>
      <td>40.463700</td>
      <td>-3.749200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>59138</td>
      <td>5476</td>
      <td>7024</td>
      <td>41.871900</td>
      <td>12.567400</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Details
pd.set_option('display.max_rows', 500)  #this allows us to see all rows. 
alldata[alldata.cluster==0]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>cluster</th>
    </tr>
    <tr>
      <th>Country/Region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>40</td>
      <td>1</td>
      <td>1</td>
      <td>33.939100</td>
      <td>67.710000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>89</td>
      <td>2</td>
      <td>2</td>
      <td>41.153300</td>
      <td>20.168300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>201</td>
      <td>17</td>
      <td>65</td>
      <td>28.033900</td>
      <td>1.659600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>113</td>
      <td>1</td>
      <td>1</td>
      <td>42.506300</td>
      <td>1.521800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>-11.202700</td>
      <td>17.873900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Antigua and Barbuda</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17.060800</td>
      <td>-61.796400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>225</td>
      <td>4</td>
      <td>3</td>
      <td>-38.416100</td>
      <td>-63.616700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Armenia</th>
      <td>194</td>
      <td>0</td>
      <td>2</td>
      <td>40.069100</td>
      <td>45.038200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>1314</td>
      <td>7</td>
      <td>88</td>
      <td>-24.502867</td>
      <td>141.055589</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>3244</td>
      <td>16</td>
      <td>9</td>
      <td>47.516200</td>
      <td>14.550100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Azerbaijan</th>
      <td>65</td>
      <td>1</td>
      <td>10</td>
      <td>40.143100</td>
      <td>47.576900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bahamas, The</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>25.034300</td>
      <td>-77.396300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bahrain</th>
      <td>332</td>
      <td>2</td>
      <td>149</td>
      <td>26.066700</td>
      <td>50.557700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bangladesh</th>
      <td>27</td>
      <td>2</td>
      <td>3</td>
      <td>23.685000</td>
      <td>90.356300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Barbados</th>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>13.193900</td>
      <td>-59.543200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Belarus</th>
      <td>76</td>
      <td>0</td>
      <td>15</td>
      <td>53.709800</td>
      <td>27.953400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>3401</td>
      <td>75</td>
      <td>263</td>
      <td>50.503900</td>
      <td>4.469900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Benin</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>9.307700</td>
      <td>2.315800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bhutan</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>27.514200</td>
      <td>90.433600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bolivia</th>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>-16.290200</td>
      <td>-63.588700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bosnia and Herzegovina</th>
      <td>126</td>
      <td>1</td>
      <td>2</td>
      <td>43.915900</td>
      <td>17.679100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>1593</td>
      <td>25</td>
      <td>2</td>
      <td>-14.235000</td>
      <td>-51.925300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Brunei</th>
      <td>88</td>
      <td>0</td>
      <td>2</td>
      <td>4.535300</td>
      <td>114.727700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bulgaria</th>
      <td>187</td>
      <td>3</td>
      <td>3</td>
      <td>42.733900</td>
      <td>25.485800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Burkina Faso</th>
      <td>75</td>
      <td>4</td>
      <td>5</td>
      <td>12.238300</td>
      <td>-1.561600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cabo Verde</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16.538800</td>
      <td>-23.041800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>84</td>
      <td>0</td>
      <td>1</td>
      <td>12.565700</td>
      <td>104.991000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cameroon</th>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>3.848000</td>
      <td>11.502100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1465</td>
      <td>21</td>
      <td>10</td>
      <td>50.993533</td>
      <td>-92.262983</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cape Verde</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.111100</td>
      <td>-23.616700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Central African Republic</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>6.611100</td>
      <td>20.939400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chad</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>15.454200</td>
      <td>18.732200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chile</th>
      <td>632</td>
      <td>1</td>
      <td>8</td>
      <td>-35.675100</td>
      <td>-71.543000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colombia</th>
      <td>231</td>
      <td>2</td>
      <td>3</td>
      <td>4.570900</td>
      <td>-74.297300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Congo (Brazzaville)</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-0.228000</td>
      <td>15.827700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Congo (Kinshasa)</th>
      <td>30</td>
      <td>1</td>
      <td>0</td>
      <td>-4.038300</td>
      <td>21.758700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Costa Rica</th>
      <td>134</td>
      <td>2</td>
      <td>2</td>
      <td>9.748900</td>
      <td>-83.753400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cote d'Ivoire</th>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>7.540000</td>
      <td>-5.547100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Croatia</th>
      <td>254</td>
      <td>1</td>
      <td>5</td>
      <td>45.100000</td>
      <td>15.200000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cruise Ship</th>
      <td>712</td>
      <td>8</td>
      <td>325</td>
      <td>35.449800</td>
      <td>139.664900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cuba</th>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>21.521800</td>
      <td>-77.781200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Cyprus</th>
      <td>95</td>
      <td>1</td>
      <td>3</td>
      <td>35.126400</td>
      <td>33.429900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Czechia</th>
      <td>1120</td>
      <td>1</td>
      <td>6</td>
      <td>49.817500</td>
      <td>15.473000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>1514</td>
      <td>13</td>
      <td>1</td>
      <td>63.287800</td>
      <td>-13.338100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Djibouti</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>11.825100</td>
      <td>42.590300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>15.415000</td>
      <td>-61.371000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Dominican Republic</th>
      <td>202</td>
      <td>3</td>
      <td>0</td>
      <td>18.735700</td>
      <td>-70.162700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>East Timor</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-8.550000</td>
      <td>125.560000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ecuador</th>
      <td>789</td>
      <td>14</td>
      <td>3</td>
      <td>-1.831200</td>
      <td>-78.183400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Egypt</th>
      <td>327</td>
      <td>14</td>
      <td>56</td>
      <td>26.820600</td>
      <td>30.802500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>El Salvador</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>13.794200</td>
      <td>-88.896500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Equatorial Guinea</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1.650800</td>
      <td>10.267900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Eritrea</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>15.179400</td>
      <td>39.782300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Estonia</th>
      <td>326</td>
      <td>0</td>
      <td>2</td>
      <td>58.595300</td>
      <td>25.013600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Eswatini</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>-26.522500</td>
      <td>31.465900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ethiopia</th>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>9.145000</td>
      <td>40.489700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>-17.713400</td>
      <td>178.065000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Finland</th>
      <td>626</td>
      <td>1</td>
      <td>10</td>
      <td>61.924100</td>
      <td>25.748200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>French Guiana</th>
      <td>18</td>
      <td>0</td>
      <td>6</td>
      <td>3.933900</td>
      <td>-53.125800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>-0.803700</td>
      <td>11.609400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Gambia, The</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.443200</td>
      <td>-15.310100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>54</td>
      <td>0</td>
      <td>3</td>
      <td>42.315400</td>
      <td>43.356900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ghana</th>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>7.946500</td>
      <td>-1.023200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Greece</th>
      <td>624</td>
      <td>15</td>
      <td>19</td>
      <td>39.074200</td>
      <td>21.824300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Greenland</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>72.000000</td>
      <td>-40.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Grenada</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>12.116500</td>
      <td>-61.679000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guadeloupe</th>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>16.265000</td>
      <td>-61.551000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guam</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>13.444300</td>
      <td>144.793700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>15.783500</td>
      <td>-90.230800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guernsey</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49.450000</td>
      <td>-2.580000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guinea</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>9.945600</td>
      <td>-9.696600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Guyana</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>4.860400</td>
      <td>-58.930200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Haiti</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>18.971200</td>
      <td>-72.285200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Holy See</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>41.902900</td>
      <td>12.453400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Honduras</th>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>15.200000</td>
      <td>-86.241900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Hungary</th>
      <td>131</td>
      <td>6</td>
      <td>16</td>
      <td>47.162500</td>
      <td>19.503300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Iceland</th>
      <td>568</td>
      <td>1</td>
      <td>36</td>
      <td>64.963100</td>
      <td>-19.020800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>396</td>
      <td>7</td>
      <td>27</td>
      <td>20.593700</td>
      <td>78.962900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Indonesia</th>
      <td>514</td>
      <td>48</td>
      <td>29</td>
      <td>-0.789300</td>
      <td>113.921300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Iraq</th>
      <td>233</td>
      <td>20</td>
      <td>57</td>
      <td>33.223200</td>
      <td>43.679300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ireland</th>
      <td>906</td>
      <td>4</td>
      <td>5</td>
      <td>53.142400</td>
      <td>-7.692100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Israel</th>
      <td>1071</td>
      <td>1</td>
      <td>37</td>
      <td>31.046100</td>
      <td>34.851600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Jamaica</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>18.109600</td>
      <td>-77.297500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>1086</td>
      <td>40</td>
      <td>235</td>
      <td>36.204800</td>
      <td>138.252900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Jersey</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49.190000</td>
      <td>-2.110000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Jordan</th>
      <td>112</td>
      <td>0</td>
      <td>1</td>
      <td>30.585200</td>
      <td>36.238400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kazakhstan</th>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>48.019600</td>
      <td>66.923700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kenya</th>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>-0.023600</td>
      <td>37.906200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Korea, South</th>
      <td>8897</td>
      <td>104</td>
      <td>2909</td>
      <td>35.907800</td>
      <td>127.766900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kosovo</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>42.602600</td>
      <td>20.903000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kuwait</th>
      <td>188</td>
      <td>0</td>
      <td>27</td>
      <td>29.311700</td>
      <td>47.481800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Kyrgyzstan</th>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>41.204400</td>
      <td>74.766100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Latvia</th>
      <td>139</td>
      <td>0</td>
      <td>1</td>
      <td>56.879600</td>
      <td>24.603200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Lebanon</th>
      <td>248</td>
      <td>4</td>
      <td>8</td>
      <td>33.854700</td>
      <td>35.862300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Liberia</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>6.428100</td>
      <td>-9.429500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Liechtenstein</th>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>47.166000</td>
      <td>9.555400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Lithuania</th>
      <td>131</td>
      <td>1</td>
      <td>1</td>
      <td>55.169400</td>
      <td>23.881300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Luxembourg</th>
      <td>798</td>
      <td>8</td>
      <td>6</td>
      <td>49.815300</td>
      <td>6.129600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Madagascar</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-18.766900</td>
      <td>46.869100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Malaysia</th>
      <td>1306</td>
      <td>10</td>
      <td>139</td>
      <td>4.210500</td>
      <td>101.975800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Maldives</th>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>3.202800</td>
      <td>73.220700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Malta</th>
      <td>90</td>
      <td>0</td>
      <td>2</td>
      <td>35.937500</td>
      <td>14.375400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Martinique</th>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>14.641500</td>
      <td>-61.024200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mauritania</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>21.007900</td>
      <td>-10.940800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mauritius</th>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>-20.348400</td>
      <td>57.552200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mayotte</th>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>-12.827500</td>
      <td>45.166200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>251</td>
      <td>2</td>
      <td>4</td>
      <td>23.634500</td>
      <td>-102.552800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Moldova</th>
      <td>94</td>
      <td>1</td>
      <td>1</td>
      <td>47.411600</td>
      <td>28.369900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Monaco</th>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>43.738400</td>
      <td>7.424600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mongolia</th>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>46.862500</td>
      <td>103.846700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Montenegro</th>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>42.708700</td>
      <td>19.374400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Morocco</th>
      <td>115</td>
      <td>4</td>
      <td>3</td>
      <td>31.791700</td>
      <td>-7.092600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mozambique</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-18.665700</td>
      <td>35.529600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Namibia</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-22.957600</td>
      <td>18.490400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Nepal</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>28.394900</td>
      <td>84.124000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>4216</td>
      <td>180</td>
      <td>2</td>
      <td>23.716450</td>
      <td>-49.180450</td>
      <td>0</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>-40.900600</td>
      <td>174.886000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Nicaragua</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>12.865400</td>
      <td>-85.207200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Niger</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>17.607800</td>
      <td>8.081700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Nigeria</th>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>9.082000</td>
      <td>8.675300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>North Macedonia</th>
      <td>114</td>
      <td>1</td>
      <td>1</td>
      <td>41.608600</td>
      <td>21.745300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>2383</td>
      <td>7</td>
      <td>1</td>
      <td>60.472000</td>
      <td>8.468900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Oman</th>
      <td>55</td>
      <td>0</td>
      <td>17</td>
      <td>21.473500</td>
      <td>55.975400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Pakistan</th>
      <td>776</td>
      <td>5</td>
      <td>5</td>
      <td>30.375300</td>
      <td>69.345100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Panama</th>
      <td>245</td>
      <td>3</td>
      <td>0</td>
      <td>8.538000</td>
      <td>-80.782100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Papua New Guinea</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-6.315000</td>
      <td>143.955500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Paraguay</th>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>-23.442500</td>
      <td>-58.443800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>363</td>
      <td>5</td>
      <td>1</td>
      <td>-9.190000</td>
      <td>-75.015200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Philippines</th>
      <td>380</td>
      <td>25</td>
      <td>17</td>
      <td>12.879700</td>
      <td>121.774000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>634</td>
      <td>7</td>
      <td>1</td>
      <td>51.919400</td>
      <td>19.145100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>1600</td>
      <td>14</td>
      <td>5</td>
      <td>39.399900</td>
      <td>-8.224500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Puerto Rico</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18.200000</td>
      <td>-66.500000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Qatar</th>
      <td>494</td>
      <td>0</td>
      <td>33</td>
      <td>25.354800</td>
      <td>51.183900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Republic of the Congo</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.440000</td>
      <td>15.556000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Reunion</th>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>-21.115100</td>
      <td>55.536400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Romania</th>
      <td>433</td>
      <td>3</td>
      <td>64</td>
      <td>45.943200</td>
      <td>24.966800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Russia</th>
      <td>367</td>
      <td>0</td>
      <td>16</td>
      <td>61.524000</td>
      <td>105.318800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rwanda</th>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>-1.940300</td>
      <td>29.873900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Saint Lucia</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>13.909400</td>
      <td>-60.978900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Saint Vincent and the Grenadines</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>12.984300</td>
      <td>-61.287200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>San Marino</th>
      <td>160</td>
      <td>20</td>
      <td>4</td>
      <td>43.942400</td>
      <td>12.457800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Saudi Arabia</th>
      <td>511</td>
      <td>0</td>
      <td>16</td>
      <td>23.885900</td>
      <td>45.079200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Senegal</th>
      <td>67</td>
      <td>0</td>
      <td>5</td>
      <td>14.497400</td>
      <td>-14.452400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Serbia</th>
      <td>222</td>
      <td>2</td>
      <td>1</td>
      <td>44.016500</td>
      <td>21.005900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Seychelles</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>-4.679600</td>
      <td>55.492000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Singapore</th>
      <td>455</td>
      <td>2</td>
      <td>144</td>
      <td>1.352100</td>
      <td>103.819800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Slovakia</th>
      <td>185</td>
      <td>1</td>
      <td>7</td>
      <td>48.669000</td>
      <td>19.699000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Slovenia</th>
      <td>414</td>
      <td>2</td>
      <td>0</td>
      <td>46.151200</td>
      <td>14.995500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Somalia</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.152100</td>
      <td>46.199600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>South Africa</th>
      <td>274</td>
      <td>0</td>
      <td>0</td>
      <td>-30.559500</td>
      <td>22.937500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sri Lanka</th>
      <td>82</td>
      <td>0</td>
      <td>3</td>
      <td>7.873100</td>
      <td>80.771800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sudan</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>12.862800</td>
      <td>30.217600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Suriname</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>3.919300</td>
      <td>-56.027800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>1934</td>
      <td>21</td>
      <td>16</td>
      <td>60.128200</td>
      <td>18.643500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>7245</td>
      <td>98</td>
      <td>131</td>
      <td>46.818200</td>
      <td>8.227500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Syria</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>34.802100</td>
      <td>38.996800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Taiwan*</th>
      <td>169</td>
      <td>2</td>
      <td>28</td>
      <td>23.700000</td>
      <td>121.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Tanzania</th>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>-6.369000</td>
      <td>34.888800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Thailand</th>
      <td>599</td>
      <td>1</td>
      <td>44</td>
      <td>15.870000</td>
      <td>100.992500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>The Bahamas</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24.250000</td>
      <td>-76.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>The Gambia</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.466700</td>
      <td>-16.600000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Timor-Leste</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-8.874200</td>
      <td>125.727500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Togo</th>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>8.619500</td>
      <td>0.824800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Trinidad and Tobago</th>
      <td>50</td>
      <td>0</td>
      <td>1</td>
      <td>10.691800</td>
      <td>-61.222500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Tunisia</th>
      <td>75</td>
      <td>3</td>
      <td>1</td>
      <td>33.886900</td>
      <td>9.537500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Turkey</th>
      <td>1236</td>
      <td>30</td>
      <td>0</td>
      <td>38.963700</td>
      <td>35.243300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Uganda</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.373300</td>
      <td>32.290300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ukraine</th>
      <td>73</td>
      <td>3</td>
      <td>1</td>
      <td>48.379400</td>
      <td>31.165600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>United Arab Emirates</th>
      <td>153</td>
      <td>2</td>
      <td>38</td>
      <td>23.424100</td>
      <td>53.847800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>5741</td>
      <td>282</td>
      <td>67</td>
      <td>37.641557</td>
      <td>-31.984943</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Uruguay</th>
      <td>135</td>
      <td>0</td>
      <td>0</td>
      <td>-32.522800</td>
      <td>-55.765800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Uzbekistan</th>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>41.377500</td>
      <td>64.585300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Venezuela</th>
      <td>70</td>
      <td>0</td>
      <td>15</td>
      <td>6.423800</td>
      <td>-66.589700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>113</td>
      <td>0</td>
      <td>17</td>
      <td>14.058300</td>
      <td>108.277200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-13.133900</td>
      <td>27.849300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-19.015400</td>
      <td>29.154900</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



Try some EDA of your own. This is an 10 point in class assignment. LMS (Section 1: In class assignment Clustering) by next Monday 3/30.  

Using the Covid-19 Clustering example,  try something different as part of the EDA.  

***Turn in ~1/2 page writeup (NOT A JUPYTER NOTEBOOK) describing what you did.***

Examples:

Try visualizing the data differently. 

Try running a different clustering algorithm. 

Try a different number of clusters. 
How would it be different if we created ratios that controlled for total population?  We tried a different clustering algorithm?

 


 




