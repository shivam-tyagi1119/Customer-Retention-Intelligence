############################################################
# 1. Import Libraries
############################################################

# Basic Libraries
import numpy as np
import pandas as pd
import pickle

# Machine Learning Models & Tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


############################################################
# 2. Global Parameters
############################################################

version = 1.0
n_splits = 5
output_file = f'model_C={version}.bin'


############################################################
# 3. Load Data Function
############################################################

def load_data(input_path):
    """Load CSV into pandas DataFrame"""
    df = pd.read_csv(input_path)
    return df

df = load_data('../Data/Loan_default.csv')


############################################################
# 4. Exploratory Data Analysis (Basic Preprocessing)
############################################################

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace('.', '_')

# Clean categorical variables
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Train-Test Split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Identify numerical and categorical features
num_features = df_full_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical = num_features[:-1] 

cat_features = [
 'education',
 'employmenttype',
 'maritalstatus',
 'hasmortgage',
 'hasdependents',
 'loanpurpose',
 'hascosigner'
]


############################################################
# 5. Training Function
############################################################

def train(df_train, y_train, C=version):

    # Convert to dictionaries for DictVectorizer
    dicts = df_train[cat_features + numerical].to_dict(orient='records')

    # Vectorization
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # Model Training
    model = RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return dv, model


############################################################
# 6. Train Final Model
############################################################

print('Training the final model...')

dv, model = train(df_full_train, df_full_train.default.values, C=version)


############################################################
# 7. Save Model
############################################################

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')
