#%%
import pandas as pd
import numpy as np
import gurobipy as gp
import SmartHouse as sh

class Solution:

    def __init__(self, thetas: list, obj: list, cost: list, discomfort: list, 
                 x: dict, date: str, instance: int, stochastic_components: pd.DataFrame, errors = None) -> None:
        
        self.obj = self.build_objective_df(thetas, obj, cost, discomfort, date, instance)

        self.x = self.build_decision_df(x, date, instance, stochastic_components)

        if errors is not None:

            self.error_df = self.build_error_df(instance, date, errors)

    def build_error_df(self, instance, date, errors):

        error_df = pd.DataFrame(errors.mean()).transpose()
        error_df['date'] = date
        error_df['instance'] = instance

        return error_df

        
    def build_objective_df(self, thetas, obj, cost, discomfort, date, instance):

        objective_df = pd.DataFrame(columns=['theta', 'objective', 'electricity cost', 'discomfort index'])
        
        for theta in thetas:

            df = pd.DataFrame({
                'dailyObjective': obj[theta],
                'electricity cost': cost[theta],
                'discomfort index': discomfort[theta]
            })

            # Add datetime
            df['time'] = pd.date_range(end=date+" 23:45", periods=len(df), freq='15T')
            df['date'] = date
            df['instance'] = instance
            df['theta'] = theta
            df['objective'] = theta*df['electricity cost'] + (1-theta)*df['discomfort index']

            # Add to dataframe with previous theta values
            objective_df = pd.concat([objective_df, df], ignore_index=True)

        return objective_df

    
    def build_decision_df(self, x: dict, date: str, instance:int, stochastic_components: pd.DataFrame):
        
        # Initialize decision dataframe
        decision_df = pd.DataFrame(columns=x[list(x.keys())[0]].keys())
        decision_df = decision_df.drop(columns=['a_nt', 'w_nt'])

        for theta in x.keys():

            # Get decision dictionary without decision a and w
            x2 = x[theta].copy()
            del x2['a_nt']
            del x2['w_nt']

            # Build dataframe for all decisions different from a and w
            df = pd.DataFrame(x2)

            # Build dataframes for decisions a and w
            df_a = pd.DataFrame(x[theta]['a_nt'])
            df_a.columns = ["a_"+i for i in df_a.columns]
            df_w = pd.DataFrame(x[theta]['w_nt'])
            df_w.columns = ["w_"+i for i in df_w.columns]

            # Build dataframes for all decisions
            df = pd.concat([df, df_a, df_w], axis=1)

            # Add datetime
            df['time'] = pd.date_range(end=date+" 23:45", periods=len(df), freq='15T')
            df['date'] = date
            df['instance'] = instance
            df['theta'] = theta

            # Add stochastic components
            df = pd.concat([df, stochastic_components.reset_index(drop=True)], axis=1)

            # Add to dataframe with previous theta values
            decision_df = pd.concat([decision_df, df], ignore_index=True)

        return decision_df


class OraclePolicy:

    def __init__(self, env: sh.SmartHouse) -> None:
        
        # Model of the system
        self.env = env   

        # Stochastic parameters
        self.stochastic_components = self.env.load_future_info()
        
        self.g = self.stochastic_components['solar'].to_numpy()           # g_t: Energy genereated from PV panels at time t
        self.l = self.stochastic_components['nonshiftable'].to_numpy()    # l_t: Unshiftable load at time t
        self.p = self.stochastic_components['p'].to_numpy()               # p_t: Buying price of electricity from the grid at time t
        self.q = self.p*self.env.gamma                               # q_t: selling price of electricity from the grid at time t

        self.stochastic_components.rename(columns={'solar': 'g_t', 'nonshiftable': 'l_t', 'p': 'p_t'}, inplace=True)

    def solve(self, theta_values):

        objectives = {}   # Objectives for each theta value
        costs = {}     # Electricity costs for each theta value
        discomforts = {}  # Discomfort index for each theta value
        x = {}            # Optimal solutions for each theta value


        for theta in theta_values:

            # Optimization model
            model = gp.Model("smart_house")
            model.Params.LogToConsole = 0

            # 3. Define decision variables

            # Grid variables
            b = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS , name = 'b')   # b_t: energy bought from the grid at time t
            s = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS, name = 's')    # s_t: energy sold to the grid at time t
            
            # Appliance variables
            a = model.addVars(self.env.N,self.env.T, vtype = gp.GRB.BINARY, name = 'a')      # a_(n,t): 1 if appliance n opereates at time t
            w = model.addVars(self.env.N,self.env.T, vtype = gp.GRB.BINARY, name = 'w')      # w_(n,t): 1 if appliance n starts to operate at time t
           
            # Battery variables           
            h =  model.addVars(self.env.T, lb=float('-inf'), vtype = gp.GRB.CONTINUOUS, name = 'h')  # h_t: energy transferred to or from the battery at time t
            e = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS, name = 'e')    # e_t: energy stored in the battery at time t

            # Objective variables
            d = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS, name = 'd')    # d_t: user discomfort at t
            c = model.addVars(self.env.T, lb=float('-inf'), vtype = gp.GRB.CONTINUOUS, name = 'c')    # c_t: electricity costs at t


            # 4. Restrictions

            # 4.1. Power balance
            for t in self.env.T:
                model.addConstr(b[t]+self.g[t]==h[t]+self.l[t]+gp.quicksum(self.env.delta*self.env.v[n]*a[n,t] for n in self.env.N)+s[t])
            
            # Upper bound of bought and sold electricity
            for t in self.env.T:
                model.addConstr(b[t] <= self.env.b_max)
                model.addConstr(s[t] <= self.env.s_max)

            # 4.2. Battery dynamics
            model.addConstr(e[0]==self.env.ebar_t+self.env.eta*h[0])
            for t in self.env.T:
                if t>0:
                    model.addConstr(e[t]==e[t-1]+self.env.eta*h[t])
                    
            # 4.3. Battery capacity and charging bounds
            for t in self.env.T:
                model.addConstr(e[t]>=self.env.e_min)
                model.addConstr(e[t]<=self.env.e_max)
                model.addConstr(h[t]>=self.env.h_min)
                model.addConstr(h[t]<=self.env.h_max)

            # 4.4. Shiftable appliances

            # Consecutive runtime
            for n in self.env.N:
                for t in self.env.T:     
                    model.addConstr(gp.quicksum(a[n,k] for k in np.arange(t, min(t+self.env.r[n]-1 + 1, self.env.T.shape[0]-1)))>=self.env.r[n]*w[n,t])

            # All appliances must run
            for n in self.env.N:
                model.addConstr(gp.quicksum(w[n,t] for t in self.env.T) == 1)

            # 4.5. User discomfort
            for t in self.env.T:
                model.addConstr(d[t] == gp.quicksum(a[n,t]*(1-self.env.pm.loc[t,n]) for n in self.env.N))

            # 4.6. Electricity costs
            for t in self.env.T:
                model.addConstr(c[t] == b[t]*self.p[t] - self.q[t]*s[t])

            # 5. Objective Function
            model.setObjective(gp.quicksum(theta*c[t]+(1-theta)*d[t] for t in self.env.T), gp.GRB.MINIMIZE)
            model.update()
            model.optimize()

            # Save results
            objectives[theta]=model.objVal
            x[theta] = {'b_t': [b[t].x for t in self.env.T],
                        's_t': [s[t].x for t in self.env.T],
                        'a_nt': [{n: a[n,t].x for n in self.env.N} for t in self.env.T],
                        'w_nt': [{n: w[n,t].x for n in self.env.N} for t in self.env.T],
                        'h_t': [h[t].x for t in self.env.T],
                        'e_t': [e[t].x for t in self.env.T],
                        'd_t': [d[t].x for t in self.env.T],
                        'c_t': [c[t].x for t in self.env.T]}
            costs[theta] = x[theta]['c_t']
            discomforts[theta] = x[theta]['d_t']

        # Return solution
        solution = Solution(thetas = theta_values, obj = objectives, cost = costs, discomfort = discomforts,
                             x = x, date = self.env.date, instance = self.env.instance, stochastic_components = self.stochastic_components)
        return solution


class DeterministicPolicy():

    def __init__(self, env: sh.SmartHouse) -> None:
        
        # Model of the system
        self.env = env

        # Future values of stochastic parameters
        self.stochastic_components = self.env.load_future_info()
        self.stochastic_components.rename(columns={'solar': 'g_t', 'nonshiftable': 'l_t', 'p': 'p_t'}, inplace=True)

        # Stochastic parameters
        self.g = env.initial_PV_forecast.copy()          # g_t: Energy genereated from PV panels at time t
        self.l = env.initial_l_forecast.copy()          # l_t: Unshiftable load at time t
        self.p = env.dayahead_prices.copy()          # p_t: Buying price of electricity from the grid at time t
        self.q = env.initial_q_forecast.copy()    # q_t: selling price of electricity from the grid at time t

    def solve(self, theta_values):

            objectives = {}   # Objectives for each theta value
            costs = {}      # Electricity costs for each theta value
            discomforts = {}  # Discomfort index for each theta value
            x = {}            # Optimal solutions for each theta value

            for theta in theta_values:

                # Optimization model
                model = gp.Model("smart_house")
                model.Params.LogToConsole = 0

                # 3. Define decision variables

                # Grid variables
                b = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS , name = 'b')   # b_t: energy bought from the grid at time t
                s = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS, name = 's')    # s_t: energy sold to the grid at time t
                
                # Appliance variables
                a = model.addVars(self.env.N,self.env.T, vtype = gp.GRB.BINARY, name = 'a')      # a_(n,t): 1 if appliance n opereates at time t
                w = model.addVars(self.env.N,self.env.T, vtype = gp.GRB.BINARY, name = 'w')      # w_(n,t): 1 if appliance n starts to operate at time t
            
                # Battery variables           
                h =  model.addVars(self.env.T, lb=float('-inf'), vtype = gp.GRB.CONTINUOUS, name = 'h')  # h_t: energy transferred to or from the battery at time t
                e = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS, name = 'e')    # e_t: energy stored in the battery at time t

                # Objective variables
                d = model.addVars(self.env.T, vtype = gp.GRB.CONTINUOUS, name = 'd')    # d_t: user discomfort at t
                c = model.addVars(self.env.T, lb=float('-inf'), vtype = gp.GRB.CONTINUOUS, name = 'c')    # c_t: electricity costs at t


                # 4. Restrictions

                # 4.1. Power balance
                for t in self.env.T:
                    model.addConstr(b[t]+self.g[t]==h[t]+self.l[t]+gp.quicksum(self.env.delta*self.env.v[n]*a[n,t] for n in self.env.N)+s[t])
                
                # Upper bound of bought and sold electricity
                for t in self.env.T:
                    model.addConstr(b[t] <= self.env.b_max)
                    model.addConstr(s[t] <= self.env.s_max)

                # 4.2. Battery dynamics
                model.addConstr(e[0]==self.env.ebar_t+self.env.eta*h[0])
                for t in self.env.T:
                    if t>0:
                        model.addConstr(e[t]==e[t-1]+self.env.eta*h[t])
                        
                # 4.3. Battery capacity and charging bounds
                for t in self.env.T:
                    model.addConstr(e[t]>=self.env.e_min)
                    model.addConstr(e[t]<=self.env.e_max)
                    model.addConstr(h[t]>=self.env.h_min)
                    model.addConstr(h[t]<=self.env.h_max)

                # 4.4. Shiftable appliances

                # Consecutive runtime
                for n in self.env.N:
                    for t in self.env.T:     
                        model.addConstr(gp.quicksum(a[n,k] for k in np.arange(t, min(t+self.env.r[n]-1 + 1, self.env.T.shape[0]-1)))>=self.env.r[n]*w[n,t])

                # All appliances must run
                for n in self.env.N:
                    model.addConstr(gp.quicksum(w[n,t] for t in self.env.T) == 1)

                # 4.5. User discomfort
                for t in self.env.T:
                    model.addConstr(d[t] == gp.quicksum(a[n,t]*(1-self.env.pm.loc[t,n]) for n in self.env.N))

                # 4.6. Electricity costs
                for t in self.env.T:
                    model.addConstr(c[t] == b[t]*self.p[t] - self.q[t]*s[t])

                # 5. Objective Function
                model.setObjective(gp.quicksum(theta*c[t]+(1-theta)*d[t] for t in self.env.T), gp.GRB.MINIMIZE)
                model.update()
                model.optimize()

                # Save results
                # objectives[theta] = model.objVal
                x[theta] = {'b_t': [b[t].x for t in self.env.T],
                            's_t': [s[t].x for t in self.env.T],
                            'a_nt': [{n: a[n,t].x for n in self.env.N} for t in self.env.T],
                            'w_nt': [{n: w[n,t].x for n in self.env.N} for t in self.env.T],
                            'h_t': [h[t].x for t in self.env.T],
                            'e_t': [e[t].x for t in self.env.T],
                            'd_t': [d[t].x for t in self.env.T],
                            'c_t': [c[t].x for t in self.env.T]}
                
                # Evaluate decisions
                self.env.theta = theta*1
                done = False
                objective_list = []
                cost_list = []
                discomfort_list= []
                t=0
                while not done:
                    
                    # Dictionary of current decisions
                    x_tt = {}
                    for key, value in x[theta].items():
                        x_tt[key] = value[t]
                    
                    obj, cost, discomfort, done = self.env.step(x_tt, update_forecasts=False)
                    objective_list.append(obj)
                    cost_list.append(cost)
                    discomfort_list.append(discomfort)
                    t += 1
                
                objectives[theta]=objective_list
                costs[theta] = cost_list
                discomforts[theta] = discomfort_list

                # Reset house
                self.env.reset_house()

            # Return solution
            solution = Solution(theta_values, objectives, costs, discomforts,
                                 x, self.env.date, self.env.instance, self.stochastic_components)
            return solution

class LookaheadPolicy():

    def __init__(self, env: sh.SmartHouse) -> None:
        
        # Model of the system
        self.env = env

        # Future values of stochastic parameters
        self.stochastic_components = self.env.load_future_info()
        self.stochastic_components.rename(columns={'solar': 'g_t', 'nonshiftable': 'l_t', 'p': 'p_t'}, inplace=True)


    def update_parameters(self) -> None:
        
        # Stochastic parameters
        self.g = self.env.fg_t.copy()          # g_t: Energy genereated from PV panels at time t
        self.l = self.env.fl_t.copy()          # l_t: Unshiftable load at time t
        self.p = self.env.fp_t.copy()          # p_t: Buying price of electricity from the grid at time t
        self.q = self.env.fq_t.copy()    # q_t: selling price of electricity from the grid at time t

    def update_errors(self, errors, p, g, l) -> np.array:

        p_mape = abs(self.stochastic_components['p_t'][-1 * p.size:] - p)/abs(self.stochastic_components['p_t'][-1 * p.size:])
        p_mse = np.sqrt((self.stochastic_components['p_t'][-1 * p.size:] - p)**2)

        g_mape = abs(self.stochastic_components['g_t'][-1 * g.size:] - g)/abs(self.stochastic_components['g_t'][-1 * g.size:])
        g_mse = np.sqrt((self.stochastic_components['g_t'][-1 * g.size:] - g)**2)

        l_mape = abs(self.stochastic_components['l_t'][-1 * l.size:] - l)/abs(self.stochastic_components['l_t'][-1 * l.size:])
        l_mse = np.sqrt((self.stochastic_components['l_t'][-1 * l.size:] - l)**2)
        
        if errors is None:

            errors = pd.DataFrame({'p_mape': p_mape,
                                    'p_rmse': p_mse,
                                    'g_mape': g_mape,
                                    'g_rmse': g_mse,
                                    'l_mape': l_mape,
                                    'l_mse': l_mse})
            
        else:

            df = pd.DataFrame({'p_mape': p_mape,
                                'p_rmse': p_mse,
                                'g_mape': g_mape,
                                'g_rmse': g_mse,
                                'l_mape': l_mape,
                                'l_mse': l_mse})
            errors = pd.concat([errors, df], ignore_index=True)

        return errors


    def solve(self, theta_values):

        objectives = {}   # Objectives for each theta value
        costs = {}      # Electricity costs for each theta value
        discomforts = {}  # Discomfort index for each theta value
        x = {}            # Optimal solutions for each theta value
        errors = None

        for theta in theta_values:
            
            # Intitialize dict to save decisions
            x[theta] = {'b_t': [],
                        's_t': [],
                        'a_nt': [],
                        'w_nt': [],
                        'h_t': [],
                        'e_t': [],
                        'd_t': [],
                        'c_t': []}
            
            # Evaluate decisions
            self.env.theta = theta*1
            done = False
            objective_list = []
            cost_list = []
            discomfort_list= []
        
            while not done:

                # Update parameters
                self.update_parameters()

                if theta == theta_values[0]:
                    # Update errors
                    errors = self.update_errors(errors = errors, p=self.p, g = self.g, l = self.l)

                # Update T set
                T = np.arange(self.env.max_t+1 - self.env.t) 

                # Optimization model
                model = gp.Model("smart_house")
                model.Params.LogToConsole = 0

                # 3. Define decision variables

                # Grid variables
                b = model.addVars(T, vtype = gp.GRB.CONTINUOUS , name = 'b')   # b_t: energy bought from the grid at time t
                s = model.addVars(T, vtype = gp.GRB.CONTINUOUS, name = 's')    # s_t: energy sold to the grid at time t
                
                # Appliance variables
                a = model.addVars(self.env.N,T, vtype = gp.GRB.BINARY, name = 'a')      # a_(n,t): 1 if appliance n opereates at time t
                w = model.addVars(self.env.N,T, vtype = gp.GRB.BINARY, name = 'w')      # w_(n,t): 1 if appliance n starts to operate at time t
            
                # Battery variables           
                h =  model.addVars(T, lb=float('-inf'), vtype = gp.GRB.CONTINUOUS, name = 'h')  # h_t: energy transferred to or from the battery at time t
                e = model.addVars(T, vtype = gp.GRB.CONTINUOUS, name = 'e')    # e_t: energy stored in the battery at time t

                # Objective variables
                d = model.addVars(T, vtype = gp.GRB.CONTINUOUS, name = 'd')    # d_t: user discomfort at t
                c = model.addVars(T, lb=float('-inf'), vtype = gp.GRB.CONTINUOUS, name = 'c')    # c_t: electricity costs at t


                # 4. Restrictions

                # 4.1. Power balance
                for t in T:
                    model.addConstr(b[t]+self.g[t]==h[t]+self.l[t]+gp.quicksum(self.env.delta*self.env.v[n]*a[n,t] for n in self.env.N)+s[t])
                
                # Upper bound of bought and sold electricity
                for t in T:
                    model.addConstr(b[t] <= self.env.b_max)
                    model.addConstr(s[t] <= self.env.s_max)

                # 4.2. Battery dynamics
                model.addConstr(e[0]==self.env.ebar_t+self.env.eta*h[0])
                for t in T:
                    if t>0:
                        model.addConstr(e[t]==e[t-1]+self.env.eta*h[t])
                        
                # 4.3. Battery capacity and charging bounds
                for t in T:
                    model.addConstr(e[t]>=self.env.e_min)
                    model.addConstr(e[t]<=self.env.e_max)
                    model.addConstr(h[t]>=self.env.h_min)
                    model.addConstr(h[t]<=self.env.h_max)

                # 4.4. Shiftable appliances

                # Consecutive runtime
                for n in self.env.N:
                    for t in T:     
                        model.addConstr(gp.quicksum(a[n,k] for k in np.arange(t, min(t+self.env.r[n]-1 + 1, T.shape[0]-1)))>=self.env.r[n]*w[n,t])

                for n in self.env.N:
                    model.addConstr(gp.quicksum(a[n,k] for k in np.arange(0, self.env.y_nt[n]-1 + 1)) >= self.env.y_nt[n])

                # All appliances must run
                for n in self.env.N:
                    model.addConstr(gp.quicksum(w[n,t] for t in T) == 1 - self.env.z_nt[n])

                # 4.5. User discomfort
                for t in T:
                    model.addConstr(d[t] == gp.quicksum(a[n,t]*(1-self.env.pm.loc[t+self.env.t,n]) for n in self.env.N))

                # 4.6. Electricity costs
                for t in T:
                    model.addConstr(c[t] == b[t]*self.p[t] - self.q[t]*s[t])

                # 5. Objective Function
                model.setObjective(gp.quicksum(theta*c[t]+(1-theta)*d[t] for t in T), gp.GRB.MINIMIZE)
                model.update()
                model.optimize()

                # Update decisions made 
                x[theta]['b_t'].append(b[0].x)
                x[theta]['s_t'].append(s[0].x)
                x[theta]['a_nt'].append({n: a[n,0].x for n in self.env.N})
                x[theta]['w_nt'].append({n: w[n,0].x for n in self.env.N})
                x[theta]['h_t'].append(h[0].x)
                x[theta]['e_t'].append(e[0].x)
                x[theta]['d_t'].append(d[0].x)
                x[theta]['c_t'].append(c[0].x)
                
                # Dictionary of current decisions
                x_tt = {}
                for key, value in x[theta].items():
                    # Get last decision stored
                    x_tt[key] = value[-1]
                
                obj, cost, discomfort, done = self.env.step(x_tt, update_forecasts=True)
                objective_list.append(obj)
                cost_list.append(cost)
                discomfort_list.append(discomfort)
        
            
            objectives[theta]=objective_list
            costs[theta] = cost_list
            discomforts[theta] = discomfort_list

            # Reset house
            self.env.reset_house()

        # Return solution
        solution = Solution(theta_values, objectives, costs, discomforts,
                                x, self.env.date, self.env.instance, self.stochastic_components, errors)
        return solution
# %%
