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


## Association Mining
The input file for the association mining analysis, had 505,564 rows by 71 columns. Python library mlxtend was used to create two models.

### Association Mining Model 1

Association mining Model 1 develops a transaction data set of force used, race, city, and if weapon/contraband was found. Minimum support was calculated by first determining the minimum support of each attribute, and setting a minimum support to be less than the maximum. It is good practice to try various minimum support values, and results compared.

Three different minimum support value (0.01, 0.02, and 0.005) for these set of transaction data was used to carry out the evaluation, and the results compared.

With a min support of 0.01, the frequent itemset was 51 with their respective support. Support for an item is the number of transactions containing that item divided by total number of transactions. For this 0.01 minimum support, the maximum number of frequent itemset occurring together is 3 and it occurred 6 times. 

The association rules were sorted by confidence, and it is showing the top 10. From the Association mining Model 1 (min support = 0.01), the rules result generally show that If suspect is from Brooklyn and force hands was used, the consequent is race_BLACK, with a 66% confidence. In general, the top three antecedents have a consequent of race_BLACK. This is not surprising since more than 50% of the SQF data was black.

A scatter plot of support vs confidence is given below. A lot of the data is concentrated on the left hand side of the plot, meaning that majority of the item sets have a minimum support less than 0.05 (higher than our defined min_support of 0.1). Out of these items majority have a confidence level less than 40%.

![am 1 scatter plot](https://user-images.githubusercontent.com/114383545/218297829-6b86a521-ee14-4958-a10d-dfc0e3acc8be.png)


Lift is the ratio of the confidence to the expected confidence of the association rule. Lift value >1 means the rule is useful, lift value <= 1 indicates not useful. 

Sorted by lift, shows use of force hands, go with use of force wall, and use of force handcuffs goes with when a weapon or contraband is found on suspect. This is expected because finding illegals on a suspect should lead to an arrest.  

Figure below is a scatter plot of support vs lift. The data is concentrated in left hand side of the plot, with approximately > 50% having a lift greater than 1. 

![am 1 scatter plot lift](https://user-images.githubusercontent.com/114383545/218297869-a3a2b9fb-558b-493c-a0f6-3540fdecc6c6.png)


### Association Mining Model 2

The Association Mining Model 2 develops a transaction data set of reasons for stop, frisk, search, and added circumstance, in addition to race, age group, and if an arrest was made. For this transaction data set, only one minimum support value (0.1) was used for the evaluation.

With a min support of 0.1, the frequent itemset was 52 with their respective support. Support for an item is the number of transactions containing that item divided by total number of transactions. For this 0.1 minimum support, the maximum number of frequent itemset occurring together is 4 and it occurred 3 times. 

The association rules were sorted by confidence, and it is showing the top 10. From the Association mining Model 2 (min support = 0.1), the rules result generally show for an age group 19 – 25 years, there is a greater than 80% confidence that the person will be stopped, and the reason for the stop will be the added circumstance of being in the area (neighbourhood). The data therefore supports that young men are generally targeted for stops based on being in a particular area.

Another common itemset that seem to go well together is reason = furtive, and added circumstance = area. 

The scatter plot is of confidence versus support, and the data points are fairly distributed over the plot, with some mild concentration at lower left hand side. Generally, majority of the association rules have a confidence greater than 65%.

![am 2 scatter plot](https://user-images.githubusercontent.com/114383545/218297899-8014cc1b-9219-48b2-b3a0-f575c2b3e446.png)

Sorting by lift, the top ten antecedents (mostly comprising race_BLACK and age group 19-25), all have consequents of been stopped for the reason furtive. 

The scatter plot of support vs lift show the data is concentrated in left hand side of the plot, with lift values greater than 1.

![am 2 scatter plot lift](https://user-images.githubusercontent.com/114383545/218297922-e5ddf199-a137-4d1a-905d-1ed3610dc1ae.png)


## Cluster Analysis
Three clustering analysis was carried out. 

### Cluster Analysis for Crime
The first used Agglomerative clustering to cluster location for crime = murder. There were 209 cases of murder in the dataset. 

To find the optimum number of clusters (best k), a ‘for loop’ is used to iterate over a range of values, to determine silhouette score for each k iterated. The results are plotted, and best k was determined to be 8. 

![cluster crime k age group](https://user-images.githubusercontent.com/114383545/218297954-0ab115c8-5735-4d05-94c9-3ebe5665d88b.png)

![cluster crime k city](https://user-images.githubusercontent.com/114383545/218297989-5f53bf59-0863-4e6a-9a6c-e86b89606f36.png)

![cluster crime k race](https://user-images.githubusercontent.com/114383545/218298009-cd60fc82-84e6-4e79-ba24-5ec1e1b13983.png)


The 8 clusters were plotted in folium map of New York City, and a scatter plot. 

![cluster crime map 8](https://user-images.githubusercontent.com/114383545/218298027-815db6cc-fbbf-42d9-97ea-a494b1b6fed1.png)

![cluster crime plot 8](https://user-images.githubusercontent.com/114383545/218298053-bee0c9bd-751a-42a0-b0d8-07d9c8d9d12a.png)

Cluster 0 was the biggest cluster with 54 cases, and Cluster 7 the smallest with 4 cases. 

Some distinct features of cluster 0:

•	more than 50% of persons arrested for murder in cluster 0 are black, 

•	the youngest is 16 years old, oldest is 62 years old, and 18 year olds had the highest count.  

•	The cluster cuts across two cities, Manhattan, and Brooklyn.

### Cluster Analysis for Stop Reasons
The second cluster analysis clustered sopped people by reason of stop using the Density-based spatial clustering of applications with noise, DBSCAN model. This method seeks to separate clusters of high density from clusters of low density and is great for clustering outliers (noise). 

The DBSCAN model does not require one to specify the number of clusters in the data a priori, as opposed to other clustering models like k-means. The model is run, fit_predict(x), and the Silhouette score = 0.75553, and number of clusters = 6.

The 6 clusters were plotted in folium map of New York City, and a scatter plot. 

![cluster stop reason map 6](https://user-images.githubusercontent.com/114383545/218298082-ec2c29ba-79f5-4a08-ad4b-576de5be3b5d.png)

![cluster stop reason plot 6](https://user-images.githubusercontent.com/114383545/218298097-b01823d7-a935-4b04-a541-9b0c10e0143c.png)

Cluster 0 is the biggest cluster with 100 cases, and Cluster 4 is the smallest cluster with 10 cases.

### Cluster Analysis for Age
A third cluster analysis was carried out on 19-year-old (using hierarchical clustering). From the data set, the highest age count was 19-year-old with a total of 28,197 instances. 

The best cluster was determined to be 5, and 9 using a ‘for’ loop range function based on city, and race respectively. Both best k was used to create models. 

##### Best k = 5

![cluster age k city](https://user-images.githubusercontent.com/114383545/218298153-50a60d54-1dcd-4ce2-a6f6-ba85dceefabd.png)

For best k = 5, cluster 0 had biggest cases at 11,542, and cluster 3 had smallest cases at 1,159. 

The 5 clusters were plotted in folium map and a scatter plot.

![cluster age map 5](https://user-images.githubusercontent.com/114383545/218298183-2bc16f6e-e2a8-4002-a014-a1339b33dc55.png)

![cluster age plot 5](https://user-images.githubusercontent.com/114383545/218298201-a4268d42-4390-409d-8f1d-8326c80fae5b.png)

##### Best K = 9

![cluster age k race](https://user-images.githubusercontent.com/114383545/218298229-19b7c61f-fa50-4ff1-bf8c-a9bfde648e0a.png)

Then when best k =9 was used, cluster 0 had biggest cases at 8,408, and cluster 8 had smallest at 947.

The 9 clusters were plotted in folium map and a scatter plot.

![cluster age map 9](https://user-images.githubusercontent.com/114383545/218298286-6032b7fc-3970-4297-aa5f-b54e8eb4f416.png)

![cluster age plot 9](https://user-images.githubusercontent.com/114383545/218298304-9b94e69d-151b-4f81-872d-0d4f085c3e2c.png)

## Predictive Modelling
This last section of the report which focuses on prediction modelling uses decision tree classifier, logistic regression, and naïve bayes to each predict if a person is armed, an arrest will be made, and if force will be used. 

Due to the large data size, KNN was not used due to its limitation in processing large datasets. A sampling could have been done for a reduce data size to enable KNN classifier to be applied, but the results may not be a true representation of the dataset.

### Armed Prediction Modelling
#### Model 1: Decision Tree Classifier for Armed Prediction

The decision tree plot is shown below. The first node 0, the root node, is the most important factor to contribute to the prediction that a person is carrying contrabands or weapons. The classifier has learned from the data that the reason ‘search others’ has a high correlation with armed.

![armed decision tree](https://user-images.githubusercontent.com/114383545/218298323-d2433a8f-2fdb-4404-8783-fee88dacd907.png)

The accuracy score is the fraction of predictions that the model predicted correctly, the precision score is focused on those predicted positive, recall measures the proportion of actual positives that got predicted as positive, and the F1 score is the harmonic mean of precision and recall. For this 4 metrics, the higher the value the better the predictive model. 

The model did very well with the train data, but only showed good accuracy for the test data, all other metrics was low.

#### Model 2: Logistic Regression Classifier for Armed Prediction

Figure below shows the results of the logistic regression classsfier. The longer the bar, the higher the importance of that attribute. The reason “search others” had the highest importance in the prediction of if a person is carrying weapons/contraband. And this result is similar to what the decision tree classifier gave us in the above section.

![armed logistic regression](https://user-images.githubusercontent.com/114383545/218298335-ea633ee0-7096-4b67-98f1-e8cb4e174366.png)

The accuracy score for both train and test data was high at over 90%. The model did not do so well in the other metrics for both train and test data.

#### Model 3: Naive Bayes Classifier for Armed Prediction

The figure below shows the coefficient of probability of the variables and how they performed with the Naïve Bayes model. 

![armed naive bayes](https://user-images.githubusercontent.com/114383545/218298348-5d7717a1-8399-4ff8-bee7-c56ca5116f64.png)

The accuracy score for both train and test data was high at over 90%, with the test data performing slightly better in both accuracy and precision score. The model did not do so well in recall and F1 score for both train and test data.

### Armed Prediction Evaluation

The armed prediction using decision tree classifier and logistic regression both identified the reason ‘search others’ as the most important variable in determining if a person stopped is armed with weapons/ contrabands.  The accuracy for the 3 models was > 90% when applied to the test data but had precision scores less than 50% for the 3 models.

### Arrest Prediction Modelling
#### Model 1: Decision Tree Classifier for Arrest Prediction

The decision tree plot is shown below. The first node 0 is the most important factor to contribute to the prediction that an arrest will be made. The classifier has learned from the data that finding contraband has a high positive correlation with an arrest being made.

![arrest decision tree](https://user-images.githubusercontent.com/114383545/218298369-79eeef40-649c-42e0-a3ea-af678e3993ef.png)

The model did well in the accuracy and precision score for both test and train data. 

#### Model 2: Logistic Regression Classifier for Arrest Prediction

Figure below shows the results of the logistic regression classsfier. The “found contraband” had the highest importance in the prediction of if a person will be arrested. And this result is similar to what the decision tree classifier gave us in the above section.

![arrest logistic regression](https://user-images.githubusercontent.com/114383545/218298386-1e222c7c-76e9-4ad9-98f1-f347f5f15e84.png)

The accuracy score for both train and test data was high at over 90% and was over 70% for the precision score.

#### Model 3: Naive Bayes Classifier for Arrest Prediction

The figure below shows the coefficient of probability of the variables and how they performed with the Naïve Bayes model. 

![arrest naive bayes](https://user-images.githubusercontent.com/114383545/218298398-40c673a4-8fcb-4823-8ae6-453b81ba24c0.png)

The accuracy score and precision score for both train and test data was high, but the train data did better than the test data in the precision metric. 

### Arrest Prediction Evaluation

For arrest prediction, the most important variable (based on both decision tree classifier and logistic regression classifier) was found contraband. This was not surprising because it is expected that an arrest will be made by an officer if weapons/contraband are found on stopped persons. The accuracy and precision scores for the 3 models were higher than 90% and 70% respectively when applied to the test data.

### Force Prediction Modelling

#### Model 1: Decision Tree Classifier for Force Prediction
The decision tree plot is shown below. The first node 0 is the most important factor to contribute to the prediction that force will be used. The classifier has learned from the data that a person frisked because of the reason “furtive” has a high positive correlation that force will be used during the stop and frisk.

![force decision tree](https://user-images.githubusercontent.com/114383545/218298410-a1d77ace-77ea-43ac-8973-721d7c133a23.png)

The model did only well in the train data. 

#### Model 2: Logistic Regression Classifier for Force Prediction

Figure below shows the results of the logistic regression classsfier. The reason “search others” had the highest importance in the prediction of if force will be used. The second longest bar is for reason “frisk furtive”, and this was the result gotten from the decision tree classifier in the above section.

![force logistic regression](https://user-images.githubusercontent.com/114383545/218298435-93d78e3a-3d76-49fe-a440-96f9099ace23.png)

The accuracy score for both train and test data was high at over 80%, but the precision score was less than 50% for both data. 

#### Model 3: Naive Bayes Classifier for Force Prediction

The figure below shows the coefficient of probability of the variables and how they performed with the Naïve Bayes model. 

![force naive bayes](https://user-images.githubusercontent.com/114383545/218298459-45943df2-6485-4b93-ba93-905be3aa7b19.png)

The accuracy score for both train and test data was high at over 80%, but the precision score was less than 50% for both data, similar to the results gotten from the Logistic Regression classifier.

### Force Prediction Evaluation

The last prediction analysis on force showed some surprising results. In this instance the decision tree and logistic regression classifier did not give the same variable as the most important. 

Decision tree model learned that the reason furtive has a higher correlation for force used, while the logistic regression classifier determined that the reason “search others” was the most important variable. The accuracy score of the 3 models was greater than 70% for the test data, but less than 50% for the precision score.

In general, all 3 models gave high accuracy scores for each prediction of armed, arrest, and force. However, the test data precision score for armed prediction, and force prediction was low.

### Model Usefulness:

Predictive models can be very useful. For example, the models used to predict if a person is armed can be very useful to officers during a stop operation. This is because the model has determined attributes that can (on their own, or when combined with other attributes) be a pointer to warn officers that the person may be carrying a weapon/contraband. Then officers can approach with the required amount of caution. 

In the implementation of the model, New York city future SQF data should be evaluated. This will serve as the test model, and the validation scores measured and benchmark against the initial model created above. 

New data can be used to fine tune the parameters of the model (hyperparameters tuning), and the model can be redeployed. 

The model should be updated annually after yearly SQF data is collected and processed.




