import pandas as pd
import numpy as np

data = pd.read_csv('data/dataset_invoices.csv')

for col in ['document_create_date', 'clear_date', 'due_in_date']:
    data[col] = pd.to_datetime(data[col], dayfirst=True, errors='coerce')

koniec = data['clear_date'].max()
data['event'] = (data['isOpen'] == 0).astype(int)

data['time_days'] = np.where(data['event'] == 1,
    (data['clear_date'] - data['document_create_date']).dt.days, 
    (koniec - data['document_create_date']).dt.days)
data = data[data['time_days'] >= 0]

data = data.dropna(subset=['time_days'])

client_counts = data['cust_number'].value_counts()
treshold = client_counts.quantile(0.90)

key_enterprises = client_counts[client_counts > treshold].index
data['segment'] = np.where(data['cust_number'].isin(key_enterprises), 'BE', 'SME')

data.to_csv('data/dataset_survival.csv', index=False)