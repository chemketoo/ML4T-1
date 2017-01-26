"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    #sort by index to get order right
    orders_df=orders_df.sort_index()
    print "ORDER BOOK", orders_df

    #convert index to datetime
    orders_df.index=pd.to_datetime(orders_df.index)

    #get start date, end date of order book
    sd=orders_df.index.values[0]
    ed = orders_df.index.values[-1]

    #get all symbols in order book
    def scrape_symbols(df):
        symbol_list=[]
        for i in range(0,(df.shape[0])):
            symbol=df.iloc[i,0]
            if symbol in symbol_list:
                pass
            else:
                symbol_list.append(symbol)
        return symbol_list

    syms=scrape_symbols(orders_df)

    #create a dataframe based on the order book that contains a column for each stock listed, plus SPY, and cash column



    #dummy values for now

    #sd = dt.datetime(2010, 1, 1)
    #ed= dt.datetime(2010, 12, 31)

    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    # add cash column for later
    prices_all['Cash']=np.ones(prices_all.shape[0])

    #duplicate price df into a units df and intialize to zero
    units_all=prices_all*0.0

    #initialize starting cash position
    units_all.iloc[0,-1]=start_val

    order=orders_df.iloc[0]

    #adjust units_all to show how stock units and cash are changing over time w/orders



    for index2, row2 in orders_df.iterrows():
        stock_name=row2[0]
        order_price = prices_all[stock_name].ix[index2]
        order_units = row2[2]
        if row2[1]=="BUY":
            pos_multplr=-1
        else:
            pos_multplr=1
        #update units_all with order
        units_all.loc[index2,stock_name]+=order_units*pos_multplr*-1
        units_all.loc[index2,"Cash"]+=order_units*order_price*pos_multplr

    print units_all.head()

    #now update units_all to be full accounting table of units over time
    for i in range(1,units_all.shape[0]):
        for j in range (0,units_all.shape[1]):
            new_val=units_all.iloc[i,j]+units_all.iloc[i-1,j]
            units_all.iloc[i,j]=new_val

    #finally get port_vals
    port_vals=prices_all*units_all

    port_vals["port_val"]=port_vals.sum(axis=1)

    port_vals["daily_returns"] = (port_vals["port_val"][1:] / port_vals["port_val"][:-1].values) - 1
    port_vals["daily_returns"][0] = 0

    #now we have the port_val by day so can calculate common statistics

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

    cr, adr, sddr, sr= compute_pf_stats(port_vals,rfr=0,sf=252)

    #update row based on orders that day



    #update port_vals to only be one column of values
    port_val=port_vals.iloc[:,-2:-1]

    print "cr, adr, sddr, sr", cr, adr, sddr, sr
    return port_val

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
