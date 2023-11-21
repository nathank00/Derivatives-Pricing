## Black-Scholes Monte-Carlo Options Simulation

import numpy as np 
import math
import time



class OptionPricing:

    def __init__(self, S0, K, T, rf, sigma, iterations):
        self.S0 = S0
        self.K = K
        self.T = T
        self.rf = rf 
        self.sigma = sigma 
        self.iterations = iterations 

    def call_option_simulation(self):

        #we have 2 columns: first with 0s, second will store the payoff
        #we need the first column of 0s: payoff function is max(S-E, 0) for call option
        option_data = np.zeros([self.iterations, 2])

        #dimensions: 1-dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        #equation for the S(t) stock price
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma**2) + self.sigma *np.sqrt(self.T) * rand)

        #we need S-E in order to calculate the max(S-K, 0)
        option_data[:,1] = stock_price - self.K 

        #average for the Monte-Carlo method
        #np.amax() returns the max(S-K, 0) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)

        #we have to use the exp(-rT) discount factor
        return np.exp(-1.0*self.rf*self.T)*average

    def put_option_simulation(self):

        #we have 2 columns: first with 0s, second will store the payoff
        #we need the first column of 0s: payoff function is max(S-E, 0) for call option
        option_data = np.zeros([self.iterations, 2])

        #dimensions: 1-dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        #equation for the S(t) stock price
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma**2) + self.sigma *np.sqrt(self.T) * rand)

        #we need S-E in order to calculate the max(K-S, 0)
        option_data[:,1] = self.K - stock_price

        #average for the Monte-Carlo method
        #np.amax() returns the max(K-S, 0) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)

        #we have to use the exp(-rT) discount factor
        return np.exp(-1.0*self.rf*self.T)*average



if __name__ == "__main__":

    import datetime as dt
    from datetime import datetime
    from datetime import date
    import yahoo_fin
    from yahoo_fin import options
    from yahoo_fin import stock_info
    import pandas_datareader
    from pandas_datareader import data as pdr

    #Obtaining Ticker and Option Type

    ticker = input("Enter Ticker: ").upper()
    price = stock_info.get_live_price(f"{ticker}")
    S0 = round(price, 2)
    print(f"\nTicker Selected: {ticker} \n", f"Price: {S0}\n")

    option = input("Call or Put? ")
    print(f"\nOption Selected: {option}")

    if option == 'put' or option == 'Put':
        chain = options.get_puts(f"{ticker}")

    elif option == 'call' or option == 'Call':
        chain = options.get_calls(f"{ticker}")

    #Obtaining a Strike Price

    print("Select a Strike Price\n", chain['Strike'])
    strike = float(input("\nSelect a Strike Price: "))
    print(f"\nStrike Price Selected: {strike}")
    K = strike

    #Obtaining Expiry Date

    dates = options.get_expiration_dates(f"{ticker}")
    print("\nSelect an Expiry Date\n", dates)
    date1 = input("\nExpiry Date (MM, DD, YY): ")

    #Calculating Days to Expiry

    date_str = date1
    date_object = datetime.strptime(date_str, '%m/%d/%y').date()
    #print(type(date_object))
    #print(date_object)

    from datetime import date
    today = date.today()
    td = date_object - today
    print(f"\nDays to Expiry: {td.days}")

    T = td.days / 365

    #Obtaining Risk-Free Rate

    r = stock_info.get_live_price("^IRX")
    rf = round(r, 2)/100

    #print(f"Risk-Free Rate: {rf}")

    #Calculating Sigma (volatility)

    from dateutil.relativedelta import relativedelta
    minusoneyear = today - relativedelta(years=1)
    #print(minusoneyear)

    df = stock_info.get_data(f"{ticker}", start_date= minusoneyear, end_date= today)
    df['close']

    log_returns = np.log(df.close/df.close.shift(1)).dropna()
    daily_std = log_returns.std()
    annualized_vol = daily_std * np.sqrt(252)
    sigma = annualized_vol
    #sigma

    iterations = int(input("\nHow Many Simulations Would You Like to Run? "))
    print(f"\nSimulations = {iterations}")



    #S0 = 191.45                #underlying stock price at t=0
    #K = 170                 #strike price
    #T = 0.16438                   #expiry
    #rf = 0.0525               #risk-free rate
    #sigma = 0.2119             #volatility of underlying stock
    #iterations = 1000000   #number of iterations in the monte-carlo simulation

    model = OptionPricing(S0, K, T, rf, sigma, iterations)

    if option == 'call' or option == 'Call': 
        print("\nProjected Call option price: ", model.call_option_simulation())
    
    if option == 'put' or option == 'Put':
        print("\nProjected Put option price: ", model.put_option_simulation())