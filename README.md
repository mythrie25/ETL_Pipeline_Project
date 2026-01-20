# ETL Pipeline Project

**Author:** Mythri  
**Tools:** Python, Pandas, NumPy, Scikit-learn  

## Project Description

This project demonstrates a **data pipeline (ETL)** process:

- **Extract:** Load a CSV dataset.  
- **Transform:** Handle missing values, scale numerical features, encode categorical features.  
- **Load:** Save the processed data to a new CSV file.  

The pipeline also splits the dataset into training and testing sets for machine learning purposes.

## Dataset

The dataset used in this project is:

- `Customer Purchase Data.csv`  
- Columns include:  
  `Number, Age, Income, Spending_Score, Membership_Years, Purchase_Frequency, Last_Purchase_Amount`  

The target column used for modeling is: `Purchase_Frequency`.

## How to Run

1. Ensure you have Python 3.x installed.  
2. Install required libraries if not already installed:

```bash
pip install pandas numpy scikit-learn
