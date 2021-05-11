# Police Data Analysis: Terry Stops in Seattle

**Repository Directory**

```
├── README.md        <-- Main README file explaining the project's purpose,
│                        methodology, and findings
│
├── data             <-- Data in CSV format
│   ├── processed    <-- Processed (combined, cleaned) data used for modeling
│   └── raw          <-- Original (immutable) data dump
│
├── images           <-- Figures used in presentation and notebooks
│
├── notebooks        <-- Jupyter Notebooks for exploration and presentation
│   └── exploratory  <-- Unpolished exploratory data analysis (EDA) notebooks
│
├── reports          <-- Generated analysis (including presentation.pdf)
│
└── src              <-- Python source code for custom functions used in project
```


## Introduction
This analysis seeks to add understanding to data regarding police interactions known as Terry stops. According to [Merriam-Webster](https://www.merriam-webster.com/legal/Terry%20stop), a Terry stop is "a stop and limited search of a person for weapons justified by a police officer's reasonable conclusion that a crime is being or about to be committed by a person who may be armed and whose responses to questioning do not dispel the officer's fear of danger to the officer or to others." This type of interaction will be analyzed in depth.

More specifically, the analysis will create a model to predict whether an infraction will be added to a subject's record -- whether that is through an arrest, referral for prosecution, citation or offense report. A criminal record can have enormous impacts on an individual's life, and whether or not a subject gets let off or not have countless consequences, including employment, child custody, adoption, driving, firearms, immigration, punishment for subsequent crimes, financial aid for college admissions, and housing, among others [(1)](1). As a result of these lasting consequences, it can be valuable to understand patterns and potential biases within the way infractions are addressed within different demographics. The analysis explores topics such as race, gender, age, location, and more. 


## Data Sources & Preparation
**Source**
The city of Seattle [provides](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8) substantial publicly available data about these encounters. There are over 47,000 records spanning a period from 2015 to 2021. It includes 23 different features, including topics such as race, gender, age, location, call type, and the final resolution of the stop -- whether it ended in arrest or citation, for example.

**Preparation**
To begin, rudimentary cleaning of the data was necessary: correcting typos/inconsistencies, interpolating missing values, etc. Since some features were extermely granular, it was necessary to bin some features into larger categories to add statistical significance downstream. 

Beyond this cleaning, feature engineering included calculating the officer's age at time of Terry stop, as well as categorizing "negative" resolutions correctly to encompass outcomes that lead to lasting criminal records. 

## Data Understanding



## Modeling



## Evaluation


## Conclusion


