
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
loans = pd.read_csv('loan_approval_dataset.csv', sep=';')

#Clean up spaces around column names
loans.columns = loans.columns.str.strip()


# Display the first 20 rows
print(loans.head(20))

# Basic dataset overview
print("\nDataset dimensions:", loans.shape)
print("\nColumn names:", loans.columns.tolist())


# Question 4a: Convert categorical variables into numerical values


for col in ['education', 'self_employed', 'loan_status']:
    loans[col] = loans[col].str.strip()

# Encode binary categorical variables
loans['education'] = loans['education'].map({
    'Not Graduate': 0,
    'Graduate': 1
})

loans['self_employed'] = loans['self_employed'].map({
    'No': 0,
    'Yes': 1
})

loans['loan_status'] = loans['loan_status'].map({
    'Rejected': 0,
    'Approved': 1
})

# Making sure that the encoding was done successfully
print(loans[['education', 'self_employed', 'loan_status']].head(10))
print("\nMissing values:\n", loans.isna().sum())

# Calculate the correlation matrix
corrmatrix = loans.corr()

print(corrmatrix)


# Question 4b: Create the correlation heatmap

plt.figure(figsize=(12, 9))

sns.heatmap(
    corrmatrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Correlation Heatmap of Loan Approval Features')
plt.tight_layout()
plt.show()
