# Predict the level of fatalities in Protest and Political violence events in India

Fatalities in a protest or a political violence event is a very unfortunate outcome. This motivated me to analyse these events to see if one can predict the fatality level (Non-fatal, Lowly fatal or highly fatal) based on the information available associated with the event. Here I present a project as an attempt to address this question, please follow through the summary to see the steps involved:


## DataSource:

The Armed Conflict Location & Event Data Project (ACLED) is a disaggregated conflict collection, analysis and crisis
mapping project. ACLED collects the dates, actors, types of violence, locations, and fatalities of all reported political violence 
#and protest events. Political violence and protest includes events that occur within civil wars and periods of instability, public protest and regime breakdown. Data collected from India during the period of 26-January-2016 to 26-February-2019
source: https://www.acleddata.com/data/
Raleigh, Clionadh, Andrew Linke, Håvard Hegre and Joakim Karlsen. (2010).“Introducing ACLED-Armed Conflict Location and Event Data.” Journal of Peace Research 47(5) 651-660.

## Data Cleaning

[Jupyter Notebook](https://github.com/ViditAg/Predict_Political_violence_Fatalities/blob/master/Data_Cleaning.ipynb)

1. Drop columns that are not relavant to the analysis and do not provide any additional information.
2. Transform relavant columns to numerical values to apply supervised learning algorithms

## Data Visualization

[Jupyter Notebook](https://github.com/ViditAg/Predict_Political_violence_Fatalities/blob/master/Data_visualization.ipynb)

Break down data into fatalities labels: Non-fatal and fatal events.

Look at the distribution of event counts in all the above 3 categories for different

1. Months
2. States
4. Scale of the source of news of the event
3. Type of events

## Model Training and Evaluation

[Jupyter Notebook](https://github.com/ViditAg/Predict_Political_violence_Fatalities/blob/master/Model_training_and_evaluation.ipynb)

Applying various supervised learning alogrithms and see which one is the best in terms of accuracy of predicting fatality label especially for fatal events.

Since the data is unbalanced as non-fatal data is much higher in number than fatal events. I employ up-sampling the minority classes in training data.

## Final results

[Jupyter Notebook](https://github.com/ViditAg/Predict_Political_violence_Fatalities/blob/master/Final_Results.ipynb)

I found by upsampling and using XG Boost Classifier, we can predict fatality label of any event with *87% accuracy* and recall 92% of the fatal events.
