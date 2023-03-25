# Project Title: Classifying Emotion in Text-Based Content
## Description: 
This project aims to develop a machine learning model for emotion classification in text-based content, which will provide a more efficient and accurate method for analyzing the emotions conveyed in digital communication.

## Installation and Usage:
To install and run the Emotion Classification project, you will need to have Python 3.10 or higher and the following libraries:
-Pyspark
-Numpy 

Please see the steps below for the installations. 

The project has three steps. First clean the data, second use cleaned data and select the best model and the last one is the tune hyper-parameters of the clean. For cleaning and feature selection has been completed in the dataPrep folder in the dataPrep.py. The code cleans and saves data models to data folder. In the main folder, we need to run dataModel.py to get the best model, and in finalModelTunning file we tune hyper-parameters of our final data model.

To run the project, follow these steps:
1-Clone the repository or download the source code.
2-Install the required libraries using the following command:
```
pip install -r requirements.txt
```
3-Navigate to the dataPrep folder and run the dataPrep.py script to clean the data and generate the necessary files.
```
python dataPrep.py
```
This script expects an input file containing a list of tweets in CSV format (3 files test, training and validation), and outputs file will be stored in the data folder in parquet format.
4- Navigate back to the main folder and run the dataModel.py script to select the best model based on the cleaned data.
```
python dataModel.py
```
5-Finally, run the finalModelTunning.py script to tune the hyper-parameters of the selected model.
```
python finalModelTunning.py
```
 
4. Code Inspiration : [trajanov- Big data analytics ](https://github.com/trajanov/BigDataAnalytics)
5. Road Map: Next I will include neural network, tensorflow library implementation and compare the existing result and lastly I am planning to use the machine learning model to predict a real world data like customer feedback and rates. 
6. List of files:
- Data Folder : saved data
- dataPrep Folder : Inculudes data preparation python file for cleaning and feature selection
- dataModel file : This python file to select the best model
- finalModelTunning file : Review and hyper-tunning the final model
- projectProposal file :  Proposed research 
- projectreport: final report 

