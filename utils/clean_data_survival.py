import pandas as pd


df = pd.read_csv('data/dataset_survival.csv')
df.drop_duplicates(inplace=True)

df['due_in_date_dt'] = pd.to_datetime(df['due_in_date'])
df['document_create_date_dt'] = pd.to_datetime(df['document_create_date'])
df['baseline_create_date_dt'] = pd.to_datetime(df['baseline_create_date'], dayfirst=True)


df['days_to_due'] = (df['due_in_date_dt'] - df['document_create_date_dt']).dt.days
df['invoice_age'] = (df['due_in_date_dt'] - df['baseline_create_date_dt']).dt.days


customer_avg_delay = df.dropna(subset=['days_late']).groupby('cust_number')['days_late'].mean()
df['avg_delay_customer'] = df['cust_number'].map(customer_avg_delay).fillna(0)

df.drop(columns=['due_in_date_dt', 'document_create_date_dt', 'baseline_create_date_dt'], inplace=True)


#agregacja rzadko wystepujacych cech do other, zeby zachowac stabilnosc modelu Coxa
terms_counts = df['cust_payment_terms'].value_counts()
rare_terms = terms_counts[terms_counts < 50].index
df.loc[df['cust_payment_terms'].isin(rare_terms), 'cust_payment_terms'] = 'OTHER'

df.to_csv('data/dataset_survclean.csv', index=False)