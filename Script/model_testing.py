#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

customer = {
    'Age': 56,
    'Income': 85994,
    'LoanAmount': 50587,
    'CreditScore': 520,
    'MonthsEmployed': 80,
    'NumCreditLines': 4,
    'InterestRate': 15.23,
    'LoanTerm': 36,
    'DTIRatio': 0.44,
    'Education': "Bachelor's",
    'EmploymentType': 'Full-time',
    'MaritalStatus': 'Divorced',
    'HasMortgage': 'Yes',
    'HasDependents': 'Yes',
    'LoanPurpose': 'Other',
    'HasCoSigner': 'Yes'
}


response = requests.post(url, json=customer).json()
print(response)