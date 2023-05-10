import numpy as np
import pandas as pd



def loadResults(folder: str, results=True, decisions=True, rewrite=False) -> list:

    
    if not rewrite:
        df_results = pd.read_csv("{}/complete_results.csv".format(folder), index_col=0)
        df_results['theta'] = df_results['theta'].astype(str)
        df_results['instance'] = df_results['instance'].astype(str)

        df_decisions = pd.read_csv("{}/complete_decisions.csv".format(folder), index_col=0)
        df_decisions['theta'] = df_decisions['theta'].astype(str)
        df_decisions['instance'] = df_decisions['instance'].astype(str)
        final_list = [df_results, df_decisions]
    else:
    
        instances = pd.read_csv('Data/instances/instances.csv', index_col=0).to_numpy().reshape(14)
        instance_dict = {val: idx for idx, val in enumerate(instances)}

        policies = ["Oracle", "Deterministic", "Lookahead", "Real"]

        final_list = []

        if results:
            for p in policies:
                for i in instances:
                
                    found_result = True
                    try:
                        df = pd.read_csv("{f}/{p}/{i}/results.csv".format(f = folder,p=p, i=i), index_col=0)
                        df['policy'] = p
                    except:
                        found_result = False
                    
                    if found_result:
                        try:
                            df_results = pd.concat([df_results, df], ignore_index=True)
                        except:
                            df_results = df.copy()
            
            # Map instance
            df_results['instance'] = df_results['instance'].map(instance_dict).astype(str)
            df_results['theta'] = df_results['theta'].astype(str)
            df_results.to_csv("{}/complete_results.csv".format(folder))
            final_list.append(df_results)


        
        if decisions:
            for p in policies:
                for i in instances:
                
                    found_decisions = True
                    try:
                        df = pd.read_csv("{f}/{p}/{i}/decisions.csv".format(f=folder, p=p, i=i), index_col=0)
                        df['policy'] = p
                    except:
                        found_decisions = False
                    
                    if found_decisions:
                        try:
                            df_decisions = pd.concat([df_decisions, df], ignore_index=True)
                        except:
                            df_decisions = df.copy()

        
            df_decisions['instance'] = df_decisions['instance'].map(instance_dict).astype(str)
            df_decisions['theta'] = df_decisions['theta'].astype(str)
            df_decisions.to_csv("{}/complete_decisions.csv".format(folder))
            final_list.append(df_decisions)

    return final_list






        