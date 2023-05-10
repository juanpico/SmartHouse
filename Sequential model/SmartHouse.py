
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

DATE = '2019-06-01'
INSTANCES = pd.read_csv('Data/instances/instances.csv', index_col=0).to_numpy().reshape(14)
INSTANCE = INSTANCES[0]


class SmartHouse:
    """
    This class models the evolution of the Smart House according to a sequential decision problem with the following components:

    State (S_t): The current state of the system. 
      ebar_t (float): Energy stored in the battery at time t
      fp_t (list): Forecast of electricity buying price for the remaining horizon made at time t
      fq_t (list): Forecast of electricity selling price for the remaining horizon made at time t
      fg_t (list): Forecast of PV energy for the remaining horizon made at time t
      fl_t (list): Forecast of nonshiftable demand for the remaining horizon made at time t
      lbar_t (list): Nonshiftable demands of the last hour at time t

      ep_t (float): Error of the electricity price forecast at time t
      fep_t (float): Forecast of the electricity buying price forecast error for time t+1 made at time t
      eg_t (float): Error of the PV energy forecast at time t
      feg_t (float): Forecast of the PV energy forecast error for time t+1 made at time t

      y_nt (int): Operation time remaining for appliance n at time t
      z_nt (bool): 1 if appliance n has already been turned on at time t

      N (list): Set of appliances
      v (dict): Power of appliances
      r (dict): Runtime of appliances
      pm (DataFrame): Preference of using appliance n at time t

    Decisions (x_t): The decisions made
       b_t (float): energy purchased from the grid at time t
       s_t (float): energy sold to the grid at time t
       r_t (float): energy transferred (r_t > 0) or from (r_t < 0) the battery at time t
       e_t (float): energy stored in the battery at time t
       w_nt (dict): Appliance n starts operating at time t
       d_t (float): Discomfort index at time t
       c_t (float): Electrity cost at time t

    Exogenous information (W_t+1):
      p_t+1 (float): Price of purchasing electricity
      g_t+1 (float): Energy generated from PV panels
      l_t+1 (float): Nonshiftable demand

    Transition function: Determines the state at t+1 given the s_t, x_t and W_t+1 with method step()

    Inmediate cost (C_t): Determines the cost incurred at time t with method inmediate_cost()

    """


    def __init__(self, date: str, instance: int, 
                 gamma: float, theta: float, ebar_t: float,
                 eta: float, e_max: float, e_min: float,
                 h_min: float, h_max: float, b_max: float,
                 s_max: float, delta: float, pv_training_days: int, 
                 demand_training_days: int, initial_forecast=True, update_forecasts=True) -> None:
        
        # Filenames
        dayahead_filename = 'Data/data_dayahead.csv'
        series_filename = "Data/instances/{n}/series_{n}.csv".format(n=instance)
        pred_params_filename = "Data/instances/{n}/prediction_params_{n}.xlsx".format(n=instance)
        appliance_params_filename = "Data/instances/{n}/appliances_{n}.csv".format(n=instance)
        pm_filename = "Data/instances/{n}/pm_{n}.csv".format(n=instance)


        # Initial parameters
        self.t = 0
        self.max_t = 95
        self.T = np.arange(self.max_t+1)     # Set of timesteps
        self.date = date
        self.instance = instance
        self.series = self.load_series(series_filename)
        self.gamma = gamma
        self.theta = theta
        self.ebar_t = ebar_t

        # Model parameters
        # Battery parameters    
        self.eta = eta            # eta: charge and discharge efficiency of the battery
        self.e_max = e_max        # e_max: Capacity of the battery
        self.e_min = e_min        # e_min: Minimum energy stored in the battery
        self.h_min = h_min        # h_min: lower bound of discharge rate
        self.h_max = h_max        # h_max: upper bound of charge rate
        
        # Grid parameters
        self.b_max = b_max        # b_max: upper bound of bought electricity
        self.s_max = s_max        # s_max: upper bound of sold electricity
        self.delta = delta        # delta: size of timestep (in hours)

        # Get initial forecasts for t=0
        if initial_forecast:
            # Initial buying price forecast
            self.dayahead_prices = self.load_dayahead_prices(dayahead_filename, date)
            self.fp_t = self.dayahead_prices.copy()

            # Initial selling price forecast
            self.initial_q_forecast = self.dayahead_prices*self.gamma
            self.fq_t = self.initial_q_forecast.copy()

            # Initial PV forecast
            self.initial_PV_forecast = self.forecast_initial_PV(series_filename, date, pv_training_days)
            self.fg_t = self.initial_PV_forecast.copy()

            # Initial nonshiftable demand forecast
            self.prediction_params = self.load_prediction_params(pred_params_filename)
            self.initial_sarima_model, self.initial_l_forecast = self.forecast_initial_nonshiftable(series_filename, date, demand_training_days)
            self.sarima_model = self.initial_sarima_model
            self.fl_t = self.initial_l_forecast.copy()
            self.lbar_t = []

            # Get exogenous information for t=0
            self.price_t, self.pv_t, self.demand_t = self.get_exogenous_info()
            self.lbar_t.append(self.demand_t)

            # Get errors and error forecasts for t=0
            self.ep_t = self.price_t - self.fp_t[0]  
            self.fep_t = self.ep_t.copy()
            self.eg_t = self.pv_t - self.fg_t[0]
            self.feg_t = self.eg_t.copy()

            # Update forecasts at t=0 with the real (exogenous) values 
            self.fp_t[0] = self.price_t*1
            self.fq_t[0] = self.fp_t[0]*gamma
            self.fg_t[0] = self.pv_t*1
            self.fl_t[0] = self.demand_t*1

        # Get appliance parameters
        self.N, self.v, self.r, self.pm = self.load_appliance_params(appliance_params_filename, pm_filename)

        # Decision memory states at t=0
        self.y_nt = {n:0 for n in self.N}
        self.z_nt = {n:0 for n in self.N}



    def load_dayahead_prices(self, dayahead_filename, date) -> np.array:
        """
        Returns the dayahead prices of electricity for a given day from a loaded csv file.

        Parameters:
            prices_filename (str): The filename of the .csv that contains the dayahead prices of electricity
            date (str): A date in '%Y-%m-%d' format
        
        Returns:
            data (list): Dayahead prices of electricity
        """
        df = pd.read_csv(dayahead_filename, parse_dates=[0], index_col=0, sep=",", decimal=".")   # load csv
        df = df.resample("15min").ffill()                                                         # resample to obtain 15min frequency
 
        try:
            data = df.loc[date]                 # filter by date
            data = data.iloc[:,0].to_numpy()
            return data
        except KeyError:
            return "Date not available in provided data: " + dayahead_filename
        
    def load_previous_PV(self, series_filename, date, pv_training_days) -> pd.DataFrame:
        """
        Returns the previous PV energy generated during the last pv_training_days prior to the date.

        Parameters:
            pv_filename (str): The filename of the .csv that contains historic PV generation
            date (str): A date in '%Y-%m-%d' format
            pv_training_days (int): Amount of days for which the data is going to be loaded
        
        Returns:
            data (DataFrame): Dataframe with the PV energy data
        """
        df = self.series[['solar']]   # load pv column from series dataframe

        try:
            date = date + ' 00:00:00'
            date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

            start_date = date - timedelta(days=pv_training_days)   # Get date pv_training_days prior
            data = df.loc[start_date:date]
            data = data.iloc[:-1]

            data = data.reset_index()   
            data.columns = ['ds', 'y']  # Required format for Prophet forecasting

            #if(data.shape[0] != pv_training_days*96):
            #    return "Not enough data in PV file to train model"
            return data
        
        except KeyError:
            return "Date not available in provided data: " + series_filename

    def forecast_initial_PV(self, series_filename, date, pv_training_days) -> np.array:
        """
        Returns the initial forecast of PV generation using Prophet

        Parameters:
            pv_filename (str): The filename of the .csv that contains the PV generation
            date (str): A date in '%Y-%m-%d' format
            pv_training_days (int): Amount of days for which the data is going to be loaded
        
        Returns:
            forecast (array): Forecast of PV generation for date
        """

        df = self.load_previous_PV(series_filename, date, pv_training_days)  # Get training data


        model = Prophet(growth='flat')     # Forecast using Prophet
        model.fit(df)
        future = model.make_future_dataframe(periods=96, freq="15min", include_history=False)
        forecast = model.predict(future)[['ds','yhat']]  # Get predictions

        forecast['yhat'] = forecast['yhat'].apply(lambda x: 0 if x < 0 else x)   # Replace negative predictions with zero
        forecast['hour'] = forecast['ds'].dt.hour
        forecast.loc[((forecast['hour']>=20)|(forecast['hour']<=4)), 'yhat'] = 0 # Replace predicions between 8pm and 4am with zero
        
        forecast = forecast['yhat'].to_numpy()

        return forecast
        
    def load_previous_nonshiftable(self, series_filename, date, demand_training_days) -> pd.DataFrame:
        """
        Returns the previous nonshiftable demand during the last demand_training_days prior to the date.

        Parameters:
            demand_filename (str): The filename of the .csv that contains the historic nonshiftable demand
            date (str): A date in '%Y-%m-%d' format
            demand_training_days (int): Amount of days for which the data is going to be loaded
        
        Returns:
            data (DataFrame): Dataframe with nonshiftable demand data
        """

        df = self.series[['nonshiftable']]   # Get nonshiftable demand from series dataframe

        try:
            date = date + ' 00:00:00'
            date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

            start_date = date - timedelta(days=demand_training_days)   # Get date demand_training_days prior
            data = df.loc[start_date:date]
            data = data.iloc[:-1]

            data = data.resample("H").mean()                           # Resample with hourly frequency for sarima forecasting

            if(data.shape[0] != demand_training_days*24):
                return "Not enough data in nonshiftable demand file to train model"
            return data
        
        except KeyError:
            return "Date not available in provided data: " + series_filename
        
    def load_prediction_params(self, pred_params_filename) -> dict:
        
        """
        Returns the prediction parameters for each stochastic component

        Parameters:
            pred_params_filename (str): Filename of the excel file that contains the parameters for price, pv and demand forecasting
        
        Returns:
            params (dict): A dictionary with the value of each parameter
        """

        params = dict()
        sheets = ['price', 'pv', 'nonshiftable']

        for sheet in sheets:
            df = pd.read_excel(pred_params_filename, sheet_name=sheet, index_col=0).iloc[:,0]
            params.update(df.to_dict())

        return params

    def forecast_initial_nonshiftable(self, series_filename, date, demand_training_days) -> list:
        """
        Returns the initial nonshiftable demand

        Parameters:
            demand_filename (str): The filename of the .csv that contains the nonshiftable demand
            date (str): A date in '%Y-%m-%d' format
            demand_training_days (int): Amount of days for which the data is going to be loaded
        
        Returns:
            model (SARIMAX): SARIMAX model for prediction containing historic nonshiftable demand
            forecast (array): Forecast of nonshiftable demand
        """

        df = self.load_previous_nonshiftable(series_filename, date, demand_training_days)   # Load training data

        # Get SARIMA parameters
        order = (self.prediction_params['p'],self.prediction_params['d'],self.prediction_params['q'])  
        seasonal_order = (self.prediction_params['sp'], self.prediction_params['sd'], self.prediction_params['sq'], self.prediction_params['s'])

        # Train model with hourly frequency
        model = sm.tsa.SARIMAX(df['nonshiftable'].asfreq("H"), order = order,
                               seasonal_order=seasonal_order, freq="H").fit(disp=0)
        
        # Get initial forecast
        forecast_dates=[df.index[-1] + DateOffset(minutes=60*x)for x in range(1,25)]
        forecast = model.get_prediction(start = forecast_dates[0], end = forecast_dates[-1]).predicted_mean
        
        # Resample forecast to get 15min frequency forecast
        forecast = forecast.resample("15min").ffill()
        ix = pd.date_range(start=forecast.index[0], end=forecast.index[-1]+DateOffset(minutes=45), freq='15min')
        forecast = forecast.reindex(ix).ffill()

        forecast = forecast.to_numpy()

        return model, forecast

    def load_series(self, series_filename) -> pd.DataFrame:

        """
        Returns dataframe with historic and future information on electricity prices, pv generation and nonshiftable demand

        Parameters:
            series_filename (str): Filname of csv containing time series data
        
        Returns:
            df (DataFrame): Dataframe containing time series data
        """

        df = pd.read_csv(series_filename, parse_dates=[0], index_col=0, sep=",", decimal=".")
        return df
    
    def get_exogenous_info(self) -> list:

        """
        Returns exogenous information W_t

        Returns:
            price_t (float): Electricity buying price at time t
            pv_t (float): PV energy generated at time t
            demand_t (float): Nonshiftable demand at time t
        """
        
        current_time = self.date + ' 00:00:00'
        current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
        current_time = current_time + timedelta(minutes=15*self.t)

        pv_t, demand_t, price_t = self.series.loc[current_time]

        return price_t, pv_t, demand_t

    def load_appliance_params(self, appliance_params_filename, pm_filename)-> list:

        """
        Returns the parameters of each appliance in the household

        Parameters:
            appliance_params_filename (str): Filename of the csv containing appliance parameters
            pm_filename (str): Filename of the csv containing the PM (preference matrix) for each appliance
        
        Returns:
            N (list): List of the shiftable appliances at the household
            v (dict): Power (kW) of each appliance
            r (dict): Runtime (in 15min timesteps) of each appliance
            pm (dataFrame): PM of the appliances
        """

        # appliance pm
        app = pd.read_csv(appliance_params_filename, sep=',', decimal=".", index_col=0)
        N = app.index.tolist()
        v = app['v'].to_dict()
        r = app['r'].to_dict()

        # preference matrix
        pm = pd.read_csv(pm_filename, sep=",", decimal=".")

        return N, v, r, pm

    def step(self, x_t: dict, update_forecasts: bool) -> list:

        """
        Steps forward in time one time step, updating the state variables and returning the inmediate cost. 
        
        This method represents the transition function in Powell's framework.

        Parameters:
            x_t (dict): Decisions made at time t

        Returns:
            inmediate_cost (float): Inmediate cost of making decision x_t at state s_t
            done (bool): True if day is over

        """

        # Check factibility of the decisions
        x_t = self.check_factibility(x_t)

        # Calculate inmediate cost
        inmediate_cost, electricity_cost, discomfort_index = self.inmediate_cost(x_t)

        # Evaluate if day is finished
        if(self.t==self.max_t):
            done = True
        else:
            done=False
        
        if not done:

            self.t = self.t+1   # Step forward in time
            self.price_t, self.pv_t, self.demand_t = self.get_exogenous_info()  # Get exogenous information for t

            self.ebar_t = x_t['e_t']


            if update_forecasts:

                # Update buying price forecast
                self.update_price_forecast()

                # Update selling price forecast
                self.fq_t = self.fp_t*self.gamma 

                # Update PV forecast
                self.update_pv_forecast()

                # Update nonshiftable demand forecast
                self.update_demand_forecast()

                # Update decision memory states
                for n in self.N:

                    # Remaning runtime y_nt
                    if(x_t['w_nt'][n]==1):                    # If appliance n is turned on
                        self.y_nt[n] = self.r[n] - 1
                    elif(x_t['w_nt'][n]==0 and self.y_nt[n]>0):  # If appliance n was turned on before and has runtime remaining
                        self.y_nt[n] = self.y_nt[n] - 1
                    elif(x_t['w_nt'][n]==0 and self.y_nt[n]==0): # If appliance n is not turned on and does not have runtime remaining
                        self.y_nt[n] = 0
                    
                    # Appliances that have been turned on
                    self.z_nt[n] = self.z_nt[n] + x_t['w_nt'][n]
            
        

        

        return inmediate_cost, electricity_cost, discomfort_index, done
    
    def check_factibility(self, x_t: dict) -> dict:

        # Power balance factibility
        dif = x_t['b_t'] + self.pv_t - (x_t['h_t']+self.demand_t + sum([self.delta*self.v[n]*x_t['a_nt'][n] for n in self.N]) + x_t['s_t'])

        if dif == 0:
            # Power balance met
            pass
        elif dif > 0:
            # Energy input higher than energy output. Must sell more energy
            x_t['s_t'] = x_t['s_t'] + dif

            if x_t['s_t'] > self.s_max:
                print("Factibility correction at instance: {i}, date: {date}, t: {t}".format(i=self.instance, date=self.date, t=self.t))
                print("s_t before: {}, s_t after: {}".format(x_t['s_t']-dif, x_t['s_t']))
        elif dif < 0:
            # Energy input lower than energy output. Must buy more energy
            x_t['b_t'] = x_t['b_t'] - dif
            if x_t['b_t'] > self.b_max:
                print("Factibility correction at instance: {i}, date: {date}, t: {t}".format(i=self.instance, date=self.date, t=self.t))
                print("b_t before: {}, b_t after: {}".format(x_t['b_t']+dif, x_t['b_t']))
        return x_t

    def inmediate_cost(self, x_t) -> float:

        """
        Returns the inmediate cost of making decision x_t

        Parameters:
            x_t (dict): Decisions made at time t
        
        Returns:
            inmediate_cost (float): inmediate cost, defined by the electricity cost and the discomfort index
        """

        electricity_cost = x_t['b_t']*self.price_t - x_t['s_t']*(self.price_t*self.gamma)
        discomfort_index = np.sum([x_t['a_nt'][n]*(1-self.pm.loc[self.t, n]) for n in self.N])
       
        inmediate_cost = self.theta*electricity_cost + (1-self.theta)*discomfort_index

        return inmediate_cost, electricity_cost, discomfort_index

    def update_price_forecast(self) -> None:
        
        """
        Updates the state variable fp_t, which contains the forecast of the buying price of electricity from the grid.
        This is done by forecasting the error of the dayahead electricity prices one step into the future using simple exponential smoothing (SES)
        and by adjusting multi-step forecast with a decay parameter.
        
        Note: The first value of this forecast is the actual value of the electricity price at time t.

        """
        # Get error and error forecast for electricity prices
        self.ep_t = self.price_t - self.dayahead_prices[self.t] 
        self.fep_t = self.prediction_params['price_alpha']*self.ep_t + (1-self.prediction_params['price_alpha'])*self.fep_t

        # Update price forcast
        price_decay = np.array([self.prediction_params['price_decay']**x for x in range(0,95-self.t)])
        self.fp_t = self.dayahead_prices[self.t+1:]+price_decay*self.fep_t

        # Append actual value of the electricity price at time t
        self.fp_t = np.append(self.price_t, self.fp_t)

    def update_pv_forecast(self) -> None:

        """
        Updates the state variable fg_t, which contains the forecast of the PV generation.
        This is done by forecasting the error of the initial Prophet forecast one step into the future using simple exponential smoothing (SES)
        and by adjusting the multi-step forecast with a decay parameter.

        The forecasts are adjusted so there are no negative forecasts and the forecast between 8pm and 5am is zero.
        
        Note: The first value of this forecast is the actual value of PV energy at time t.

        """
        
        # Get error and error forecast for PV energy
        self.eg_t = self.pv_t - self.initial_PV_forecast[self.t]
        self.feg_t = self.prediction_params['pv_alpha']*self.eg_t + (1-self.prediction_params['pv_alpha'])*self.feg_t

        # Update PV energy forcast
        pv_decay = np.array([self.prediction_params['pv_decay']**x for x in range(0,95-self.t)])
        self.fg_t = self.initial_PV_forecast[self.t+1:]+pv_decay*self.feg_t

        self.fg_t[self.fg_t<0] = 0    # Replace negative predictions with zero
        
        # Replace predictions before 5am and after 8pm with zero 
        if(self.t < 19):
            self.fg_t[:19-self.t] = 0
            self.fg_t[-16:]=0
        elif(self.t>=19 and self.t<79):
            self.fg_t[-16:]=0
        else:
            self.fg_t=np.array([0 for i in self.fg_t])

        # Append the actual value of PV energy at time t
        self.fg_t = np.append(self.pv_t, self.fg_t)

    def update_demand_forecast(self) -> None:

        """
        Updates the state variable fl_t, which contains the forecast of the nonshiftable demand within the household.
        This is done by forecasting with a SARIMA model with hourly frequency. Because the model only forecasts the average 15min demand every hour,
        this forecast is only done every hour (or every 4 timesteps). Then, the forecast is resampled to obtain a forecast with 15min frequency.
        
        Note: The first value of this forecast is the actual value of nonshiftable demand at t.

        """

        if self.t%4==0: # Forecasting every hour

            # Get current hour
            current_time = self.date + ' 00:00:00'
            current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
            current_time = current_time + timedelta(minutes=15*self.t)
            current_hour = current_time.hour
            
            # Get average nonshiftable demand of the last hour
            new_index = pd.date_range(start=self.date+' 0{}:00:00'.format(current_hour-1), periods=2, freq='H')[0:1]
            new_mean_demand = np.mean(self.lbar_t)
            new_observations = pd.Series([new_mean_demand], index=new_index)

            # Update SARIMA model
            self.sarima_model = self.sarima_model.extend(new_observations)

            # Get updated forecast
            forecast_dates=[current_time + DateOffset(minutes=60*x)for x in range(0,25-current_hour-1)]
            forecast = self.sarima_model.get_prediction(start = forecast_dates[0], end = forecast_dates[-1]).predicted_mean

            # Resample the forecast to obtain 15min frequency
            forecast = forecast.resample("15min").ffill()
            ix = pd.date_range(start=forecast.index[0], end=forecast.index[-1]+DateOffset(minutes=45), freq='15min')
            forecast = forecast.reindex(ix).ffill()
            forecast = forecast.to_numpy()

            # Replace first forecast with actual nonshiftable demand at time t and update state variable
            forecast[0] = self.demand_t*1
            self.fl_t = forecast.copy()

            # Reset the nonshiftable demands of the last hour
            self.lbar_t=[]

        # Add new demand to the list
        self.lbar_t.append(self.demand_t)

    def load_future_info(self) -> pd.DataFrame:

        """
        Return stochastic data for the current day

        Returns:
            data (DataFrame): Contains electricity prices, nonshiftable demand and PV generation for the entire day
        """

        data = self.series.loc[self.date]

        return data
    
    def reset_house(self, initial_forecast = True) -> None:

        # Initial parameters
        self.t = 0

        # Get initial forecasts for t=0
        if initial_forecast:
            # Initial buying price forecast
            self.fp_t = self.dayahead_prices.copy()

            # Initial selling price forecast
            self.fq_t = self.initial_q_forecast.copy()

            # Initial PV forecast
            self.fg_t = self.initial_PV_forecast.copy()

            # Initial nonshiftable demand forecast
            self.sarima_model, self.fl_t = self.initial_sarima_model, self.initial_l_forecast.copy()
            self.lbar_t = []

            # Get exogenous information for t=0
            self.price_t, self.pv_t, self.demand_t = self.get_exogenous_info()
            self.lbar_t.append(self.demand_t)

            # Get errors and error forecasts for t=0
            self.ep_t = self.price_t - self.fp_t[0]  
            self.fep_t = self.ep_t.copy()
            self.eg_t = self.pv_t - self.fg_t[0]
            self.feg_t = self.eg_t.copy()

            # Update forecasts at t=0 with the real (exogenous) values 
            self.fp_t[0] = self.price_t*1
            self.fq_t[0] = self.fp_t[0]*self.gamma
            self.fg_t[0] = self.pv_t*1
            self.fl_t[0] = self.demand_t*1


        # Decision memory states at t=0
        self.y_nt = {n:0 for n in self.N}
        self.z_nt = {n:0 for n in self.N}



#house = SmartHouse(date=DATE, 
#                   instance = INSTANCE)



