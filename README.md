# Predict_Political_violence_Fatalities

Overview:

## DataSource:

The Armed Conflict Location & Event Data Project (ACLED) is a disaggregated conflict collection, analysis and crisis
mapping project. ACLED collects the dates, actors, types of violence, locations, and fatalities of all reported political violence 
#and protest events. Political violence and protest includes events that occur within civil wars and periods of instability, public protest and regime breakdown. Data collected from India during the period of 26-January-2016 to 26-February-2019
source: https://www.acleddata.com/data/
Raleigh, Clionadh, Andrew Linke, Håvard Hegre and Joakim Karlsen. (2010).“Introducing ACLED-Armed Conflict Location and Event Data.” Journal of Peace Research 47(5) 651-660.

## Data Cleaning

1. Drop columns that are not relavant to the analysis and do not provide any additional information.
2. Transform relavant columns to numerical values to apply supervised learning algorithms

## Exploratory Analysis

Break down data into Non-fatal, low fatalities (<5) and high fatalities(>=5).

Look at the distribution of event counts in all the above 3 categories for different

1. Months
2. States
3. Type of events
