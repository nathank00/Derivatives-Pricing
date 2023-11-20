## Black-Scholes Monte-Carlo Options Simulation

import numpy as np 
import math
import time



class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
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

        #we need S-E in order to calculate the max(S-E, 0)
        option_data[:,1] = stock_price - self.E 

        #average for the Monte-Carlo method
        #np.amax() returns the max(S-E, 0) according to the formula
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

        #we need S-E in order to calculate the max(E-S, 0)
        option_data[:,1] = self.E - stock_price

        #average for the Monte-Carlo method
        #np.amax() returns the max(E-S, 0) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)

        #we have to use the exp(-rT) discount factor
        return np.exp(-1.0*self.rf*self.T)*average



if __name__ == "__main__":

    S0 = 100                #underlying stock price at t=0
    E = 100                 #strike price
    T = 1                   #expiry
    rf = 0.05               #risk-free rate
    sigma = 0.2             #volatility of underlying stock
    iterations = 10000000   #number of iterations in the monte-carlo simulation

    model = OptionsPricing(S0, E, T, rf, sigma, iterations)
    print("Monte-Carlo Call option price: ", model.call_option_simulation())
    print("Monte-Carlo Put option price: ", model.put_option_simulation())