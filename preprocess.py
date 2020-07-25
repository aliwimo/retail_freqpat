# importing modules
import pandas as pd
import numpy as np


def regenerate_new_datafile():
    # reading dataset file
    data_frame = pd.read_excel('Online Retail.xlsx')

    # remove transactions without items.
    data_frame['StockCode'].replace('', np.nan, inplace=True)
    data_frame.dropna(subset=['StockCode'], inplace=True)

    # select only three columns that will be used later
    data_frame = data_frame[['InvoiceNo', 'StockCode', 'Description']]

    # group by same transaction ID
    data_series = data_frame.groupby('InvoiceNo')['StockCode'].unique()
    for index in range(len(data_series.values)):
        if "\'" in str(data_series.values[index]):
            data_series.values[index] = str(data_series.values[index]).replace('\'', '')

    # write processed data to new excel file
    data_series.to_excel('processed_dataset.xlsx')


regenerate_new_datafile()