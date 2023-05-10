#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import SmartHouse as sh
import Policies
from tqdm.auto import tqdm
import os

# Define instances
INSTANCES = pd.read_csv('Data/instances/instances.csv', index_col=0).to_numpy().reshape(14)

# Define dates
START_DATE = datetime(2019, 8, 1)
END_DATE = datetime(2019, 8, 31)
df_dates = pd.DataFrame([{"parameter": "start date", "value": str(START_DATE)},
            {"parameter": "end date", "value": str(END_DATE)}])

date_list = []
date = START_DATE
while date <= END_DATE:
    date_list.append(date.strftime('%Y-%m-%d'))
    date += timedelta(days=1)

# Define parameters
param_df = pd.read_excel('Params/model_params.xlsx', header=0)
model_params = dict(zip(param_df.iloc[:, 0], param_df.iloc[:, 1]))

# Update parameter df with start and end date of evaluation
param_df = pd.concat([param_df, df_dates], ignore_index=True)

# Define columns of results dataframe
columns = ['instance', 'theta', 'date', 'objective', 'electricity cost', 'discomfort index']

for i in tqdm(INSTANCES):

    results = pd.DataFrame(columns=columns)

    if i == 1240: # Instance not currently working
        continue

    first_date = True
    for date in date_list:

        # Start smart house environment
        #house = sh.SmartHouse(date=date, instance=i, theta=0.01, **model_params)

        # Get real prices
        series_filename = "Data/instances/{n}/series_{n}.csv".format(n=i)
        series = pd.read_csv(series_filename, parse_dates=[0], index_col=0, sep=",", decimal=".")
        prices = series.loc[date, ['p']]
        prices['q'] = prices['p']*model_params['gamma']

        real_filename = "Data/instances/{n}/real_{n}.csv".format(n=i)
        real = pd.read_csv(real_filename, parse_dates=[0], index_col=0, sep=",", decimal=".")
        real = real.loc[date]

        electricity_cost = real['b']*prices['p'] - real['s']*prices['q']

        theta_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

        df = pd.DataFrame({'instance': i,
                           'theta': theta_values[0],
                           'date': date,
                           'electricity cost': electricity_cost,
                           'discomfort index': 0,
                           'time': real.index})
        
        for theta in theta_values[1:]:

            df2 = pd.DataFrame({'instance': i,
                           'theta': theta,
                           'date': date,
                           'electricity cost': electricity_cost,
                           'discomfort index': 0,
                           'time': real.index})
            df = pd.concat([df, df2], ignore_index=True)
        
        df['objective'] = df['theta']*df['electricity cost'] + (1-df['theta'])*df['discomfort index']
        
        results = pd.concat([results, df], ignore_index=True)

        first_date = False
    
    # Save results
    outdir = 'Results/Real/{}'.format(i)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    results.to_csv("Results/Real/{}/results.csv".format(i))

# %%
