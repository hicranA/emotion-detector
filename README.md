# Project Title: Classifying Emotion in Text-Based Content
## Description: 
This project aims to develop a machine learning model for emotion classification in text-based content, which will provide a more efficient and accurate method for analyzing the emotions conveyed in digital communication.

## Installation and Usage:
To install and run the Emotion Classification project, you will need to have Python 3.10 or higher and the following libraries:

* pyspark
* numpy 
* spacy

Please see the steps below for the installations. 

The project has three steps. First clean the data, second use cleaned data and select the best model and the last one is the tune hyper-parameters of the clean. For cleaning and feature selection has been completed in the dataPrep folder in the dataPrep.py. The code cleans and saves data models to data folder. In the main folder, we need to run dataModel.py to get the best model, and in finalModelTunning file we tune hyper-parameters of our final data model.

To run the project, follow these steps:

1. Clone the repository or download the source code.

2. Install the required libraries using the following command:
```
pip install -r requirements.txt
```
3. Navigate to the dataPrep folder and run the dataPrep.py script to clean the data and generate the necessary files.

```
python dataPrep.py
```
This script expects an input file containing a list of tweets in CSV format (3 files test, training and validation), and outputs file will be stored in the data folder in parquet format.

4. Navigate back to the main folder and run the dataModel.py script to select the best model based on the cleaned data.

```
python dataModel.py
```
5. Finally, run the finalModelTunning.py script to tune the hyper-parameters of the selected model.

```
python finalModelTunning.py
```
## Results:
In this project, I aimed to perform sentiment analysis on Twitter data using different text preprocessing and feature selection techniques, with the goal of achieving high accuracy in classifying emotions in tweets. I found that different text preprocessing techniques had varying effects on the accuracy of the sentiment analysis models, with some techniques showing more effectiveness than others. Interestingly, I found that cleaning the data did not necessarily improve the accuracy of the model. Additionally, feature selection using chi-square performed better than not limiting feature size, indicating that selecting important features can improve accuracy in sentiment analysis models. Ultimately, I achieved an F1 accuracy of 85% with 86% precision and recall using an SVM model, which achieved the best performance on multi-class sentiment analysis.

## Limitations:
The final model still suffers of bias because of an unbalanced data. The project future roadmap includes, using different techniques to overcome current bias and improve overall accuracy. 

## Acknowledgments:
I would like to thank Professor Dimitar Trajanov and my facilatators in MET CS 777 (Big Data Analytics) class for their help. I learned a lot in a short amount of time. I am also grateful to my family and friends for their support and encouragement throughout this project. 

## Code Inspiration : 
[trajanov- Big data analytics ](https://github.com/trajanov/BigDataAnalytics)

## Road Map: 
Next I will include neural network, tensorflow library implementation and compare the existing result and lastly I am planning to use the machine learning model to predict a real world data like customer feedback and rates. 

## License:
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You are free to share, copy, and redistribute this project, as well as adapt it for non-commercial purposes, as long as you give credit to the original author(s) and include a link to the license.For more information about the CC BY-NC 4.0 license, please see [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).

## List of files:
- Data Folder : saved data
- dataPrep Folder : Includes data preparation python file for cleaning and feature selection
- projectDoc Folder: Includes project proposal, project report and project presentation files. 
- dataModel file : This python file to select the best model
- finalModelTunning file : Review and hyper-tunning the final model


