# New York City Stop Question Frisk (SQF) Data Mining Analysis

#### _A Machine Learning Project using Python scikit, mlxtend, pandas, seaborn, and matplotlib libraries._

This project analyzes the 2012 SQF datasets (>500,000 records). The analysis is divided into **four sections**: 

**1. Exploratory Data Analysis (of the dataset Attributes),** 

**2. Association Mining of frequent attributes that occur together,** 

**3. Cluster analysis,** and 

**4. Predictive modeling of three variables.**

## Exploratory Data Analysis (EDA)

### Business Understanding
New York city like any other highly populated city has its fair share of crime. The City’s police department is in the business of protection of life and property, and to effectively do this, it would make better business sense to proactively stop a crime before it happens, rather than employ resources to react to a crime that has already happened. This was the thinking behind the SQF policy, which at the time of inception in New York city, was already being implemented in other states of the U.S, and was showing some promise in those states.

### Data Understanding
New York City Police Department (NYPD) keeps a database of the Stop, Question and Frisk program.  The database contains records from 2003 to 2021, stored as csv files. This data mining report uses only data from year 2012. The 2012 SQF database contained 112 attributes (features) and 532,910 records. 
The attributes contained information on stopped persons demographics (age, height, weight, race…), officer/precinct data (precinct code, id, sector, post…), stopped location data (x coordinates, y coordinates, state, zip code…), stop date and time data (date, time, year…), stop reason, frisk reason, search reason, added circumstance reason, use of force data, crime details data, and information on contraband/weapons found. 

### Data Preparation
The data preparation steps included creating new columns (height, age group, latitude, longitude) and merging of others (date and time to give datetime). Outliers were noticed in some columns (age, height, weight), and appropriate assumptions was made to clean the data.

### EDA Visuals
Various python functions was used to evaluate the data, identify missing values, duplicates, and outliers, carry out some data cleansing, and provide key statistics of the data sets through tables and visuals.

Suspect Demographics: 
Females stopped in 2012 constituted approximately 10% of the data. 
Even at such a low percentage it might be worrisome to some, that women could be considered suspects and stopped, searched, and frisked right in the city streets.

![gender](https://user-images.githubusercontent.com/114383545/218294664-9ebb1cf1-9f21-4c46-bcf3-6ff0bce1359b.png)

The data distribution for the build of stopped persons follows a normal distribution of the population, with more than half of the stopped individuals classifed as medium built accounting for over 280,000 stops.

![build](https://user-images.githubusercontent.com/114383545/218294700-850fe63c-e29d-4de7-8eb8-749887d77bf7.png)


The race distribution of stopped suspects shows ‘Blacks’ made up more than 50% of the data (> 250,000 stops). This is unsettling because when compared to the population distribution of New York city in 2012, the black population was about 25% and white population 44%. This lends support to the lawsuit by NYCLU that the NYPD’s SQF program disproportionately target, stop, question, and frisk people of color, especially young blacks.

![race](https://user-images.githubusercontent.com/114383545/218294717-830bf4bc-1c1f-4769-ac17-ae15faa9f483.png)


The age group distribution shows the highest count of persons stopped to be between 19-25 years old, contributing to > 160,000 stops. 

![age group](https://user-images.githubusercontent.com/114383545/218294739-87874e9a-69c8-464b-8356-1589115e20ca.png)


From the chart on breakdown of the SQF by month, the first months of the year showed the highest stops in the 2012 data with a dip from June. This is contray to many crime data were its expected that criminal activities increases during the summer months and reduce during the winter period. 
This drop from June can be linked to the movement against the SQF that was raised in early 2012 that SQF was racially based, targeting young black males. This resulted in various lawsuits and litigations, and a U.S. District Court Judge, Shira Scheindlin finally ruled in 2013, that the NYPD deliberately violated the civil rights of tens of thousands of New Yorkers, and the SQF policy was in need of reforms and independent monitoring. The criticism caused Michael Bloomberg, the NYC Mayor of New York (2002 to 2013) to somewhat apologize and distance himself from the SQF policy, when he became a Democratic presedential candidate.

![month](https://user-images.githubusercontent.com/114383545/218294756-280ad569-0c80-4b2a-b572-f7713d5557f0.png)


The break down by weekday shows more stops occurring weekend (Friday and Saturday) and ramping down Sunday and Monday.

![weekday](https://user-images.githubusercontent.com/114383545/218294781-85b6475c-cadc-48ec-b437-259869505cd4.png)


The breakdown by hour shows an expected distribution of stops with the peak in the late evenings (7pm to 10pm), and the least number of stops in the early hours of the morning (3am – 8am), when its expected people are at home, and asleep.

![hour](https://user-images.githubusercontent.com/114383545/218294802-c3eabb7c-5623-4d70-aa55-c40b17b9dd8e.png)


#### Crime: 
Crime details is an important attribute in the data set, and the chart shows the top 10 crime that the SQF suspects were arrested and charged for. Criminal possession of weapon (CPW) topped the list.

![top crime](https://user-images.githubusercontent.com/114383545/218294815-d6fc536a-872e-4a85-8763-95c9a7f13272.png)


#### Force Used: 
The chart below shows the count of each type of force used by an officer on SQF suspects, with use of hands on suspect contributing to over 70,000 stops, and use of handcuffs to 20,000. 

![force used](https://user-images.githubusercontent.com/114383545/218294832-3e4f15a4-f7e4-4b48-9ef5-2f4db4661dad.png)

Figure below shows the number of force an officer used per stop. For over 400,000 stops, no force was used. 

![no of force](https://user-images.githubusercontent.com/114383545/218294887-cc9b7933-c2ef-426a-ab2a-6b8760e09e6d.png)

However, for over 30,000 stops, officers felt compelled to use force in defense of self. 

![force reason](https://user-images.githubusercontent.com/114383545/218294918-f0654e07-d02c-49de-9f62-5feb0f75e750.png)



#### Arms/Contraband Found:
In visual below, more than 8,000 suspects were found with contraband, while those found with all other weapons combined were approximately 8,000 as well. Interesting from chart above on Top 10 crimes, the count of those arrested for criminal possession of controlled substance (i.e., contraband) and criminal possession of weapons was way higher than these numbers. 

![found](https://user-images.githubusercontent.com/114383545/218294938-a05be747-f151-41fd-911e-3f3e2998ea33.png)


#### SQF Reasons:
The reasons for Stop, Question, and Frisk (SQF) contributed to 33 attributes in the SQF data. This was plotted based on stop, search, frisk, and added circumstance reasons, and is shown below.

A large percentage of those stopped was based on 2 categories. One is the reason, “furtive”, accounting for more than 50% for stop reasons and secondly, the added circumstance of “area” accounting for more than 50% of stops (>300,000). 
The definition of furtive is attempting to avoid notice or attention. What constitutes ‘attempting to avoid attention’? Was there a clear checklist given to officers to define criteria that will warrant a ‘furtive’ stop, or was this ‘a general by the book’ reason given by officers to cover racially targeted stops?

The second category, which is the added circumstance of area, could be interpreted as suspects were stopped because they were in a ‘particular’ neighbourhood. This could lend credence to the opinion that some neighbourhoods were targeted by officers for the SQF operations.

![frisk reason](https://user-images.githubusercontent.com/114383545/218294969-47b2e65d-42e7-4227-bb1f-f5a0acab75b1.png)

![stop reason](https://user-images.githubusercontent.com/114383545/218294985-fadc3827-a61f-457b-916d-95ba07bd1e9a.png)

![added](https://user-images.githubusercontent.com/114383545/218295003-5f0b0384-1c68-43d8-ab00-6febc63d220e.png)

![search](https://user-images.githubusercontent.com/114383545/218295014-6a473e77-f678-45cf-ba34-3324f2ff4a05.png)
