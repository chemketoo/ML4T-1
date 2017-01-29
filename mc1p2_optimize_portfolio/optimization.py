"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy.optimize import minimize
from util import get_data, plot_data



# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd, ed, syms, gen_plot):

    #including compute pf_stats and assess portfolio in body of optimize_portfolio to meet class reqs
    def compute_pf_stats(df, rfr, sf):
        # Get portfolio statistics (note: std_daily_ret = volatility)
        # code for stats
        cr = (df.ix[-1, -2] - df.ix[0, -2]) / df.ix[0, -2]

        # adr
        adr = df["daily_returns"][1:].mean()

        # sddr, std deviation of daily returns
        sddr = df["daily_returns"][1:].std()

        # Sharpe Ratio
        sr = (sf ** (1.0 / 2.0) * (adr - rfr)) / sddr

        # Compare daily portfolio value with SPY using a normalized plot

        return cr, adr, sddr, sr

    def assess_portfolio(sd, ed, syms, allocs, sv, gen_plot, rfr=0.0, sf=252.0, ):

        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(sd, ed)
        prices_all = get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later

        # forward fill and backward fill nans in the df
        prices = prices.fillna(method="ffill")
        prices = prices.fillna(method="bfill")

        # Get daily portfolio value
        # First normalize allocations
        # add code here to compute daily portfolio values
        # normalize the data
        prices_norm = prices / prices.ix[0, :]

        # normalize SPY data
        prices_SPY_norm = prices_SPY / prices_SPY.ix[0]

        for i, alloc in enumerate(allocs):
            prices_norm.ix[:, [i]] = prices_norm.ix[:, [i]] * alloc

        prices_norm["port_val"] = prices_norm.sum(axis=1)

        # multiply by sv to get comparison
        port_val = prices_norm["port_val"] * sv
        SPY_val = prices_SPY_norm * sv

        # calculate ev
        ev = port_val.iloc[-1]

        prices_norm["daily_returns"] = (prices_norm["port_val"][1:] / prices_norm["port_val"][:-1].values) - 1
        prices_norm["daily_returns"][0] = 0

        cr, adr, sddr, sr = compute_pf_stats(prices_norm, rfr, sf)

        # NOTE gen_plot currently uses port_val and port_val_SPY-- might make sense to use main df going forward
        if gen_plot:
            # create plot and save

            df_temp = pd.concat([port_val, SPY_val], keys=['Portfolio', 'SPY'], axis=1)
            df_temp.plot()
            plt.savefig('comparison_optimal.png')

        return cr, adr, sddr, sr, ev
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    # allocs = np.asarray([0.2, 0.2, 0.3, 0.3, 0.0]) # add code here to find the allocations
    # cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    # forward fill and backward fill nans in the df
    prices = prices.fillna(method="ffill")
    prices = prices.fillna(method="bfill")

    # Get daily portfolio value
    # First normalize allocations
    # add code here to compute daily portfolio values
    # normalize the data

    def neg_sharpe_ratio(allocs, df=prices, rfr=0.0, sf=252.0):
        df = df / df.ix[0, :]
        for i, alloc in enumerate(allocs):
            df.ix[:, [i]] = df.ix[:, [i]] * alloc
        df["port_val"] = df.sum(axis=1)
        df["daily_returns"] = (df["port_val"][1:] / df["port_val"][:-1].values) - 1
        df["daily_returns"][0] = 0
        cr = (df.ix[-1, -2] - df.ix[0, -2]) / df.ix[0, -2]
        adr = df["daily_returns"][1:].mean()
        sddr = df["daily_returns"][1:].std()
        neg_sr = -1*(sf ** (1.0 / 2.0) * (adr - rfr)) / sddr
        return neg_sr

    init_vals=np.ones(len(syms))/len(syms)
    constraints= ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bounds=tuple((0,1) for x in init_vals)
    res=minimize(neg_sharpe_ratio,init_vals,method="SLSQP",bounds=bounds, constraints=constraints)
    allocs=res.x

    #use resulting allocs for portfolio

    #calculate key statistics for optimal portfolio

    cr, adr, sddr, sr, ev= assess_portfolio(sd, ed, syms, allocs,1,gen_plot, rfr=0.0, sf=252.0)

    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    print "TEST1"
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot=False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

    print "TEST2"
    start_date = dt.datetime(2004, 1, 1)
    end_date = dt.datetime(2006, 1, 1)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

    print "PLOT SOLN" #Start Date: 2008-01-01, End Date: 2009-12-31, Symbols: ['IBM', 'X', 'HNZ', 'XOM', 'GLD']"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbols = ['IBM', 'X', 'HNZ', 'XOM', 'GLD']
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
