#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import SmartHouse as sh
import Policies
from tqdm.auto import tqdm
import os
import time


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

# Execution time list
times = np.array([])

for i in tqdm(INSTANCES):

    # get the start time
    st = time.time()

    results = pd.DataFrame(columns=columns)

    if i == 1240: # Instance not currently working
        continue

    first_date = True
    for date in date_list:

        # Start smart house environment
        house = sh.SmartHouse(date=date, instance=i, theta=0.01, **model_params, initial_forecast=False)

        # Initialize policy
        policy = Policies.OraclePolicy(house)
        theta_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        solutions = policy.solve(theta_values)

        # Get objective values
        objectives = solutions.obj
        results = pd.concat([results, objectives], ignore_index=True)

        # Get optimal decisions
        if first_date:
            decision_df = solutions.x
        else:
            decision_df = pd.concat([decision_df, solutions.x], ignore_index=True)

        first_date = False
    
    # get the end time
    et = time.time()

    # get execution time
    exec_time = et - st
    times = np.append(times,exec_time)

    # Save results
    outdir = 'Final Results/Oracle/{}'.format(i)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #results.to_csv("Final Results/Oracle/{}/results.csv".format(i))
    #decision_df.to_csv("Final Results/Oracle/{}/decisions.csv".format(i))
    #param_df.to_csv("Final Results/Oracle/{}/model_params.csv".format(i))

# Save times
times = times*(1/len(date_list))*(1/len(theta_values))
np.savetxt("Final Results/Oracle/execution_time.csv", times, delimiter=",")

# %%
