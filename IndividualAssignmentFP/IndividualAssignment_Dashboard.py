
###############################################################################
# FINANCIAL DASHBOARD 1 - v1
###############################################################################

#==============================================================================
# Initiating
#==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

#==============================================================================
# Tab 1
#==============================================================================

def tab1():
    
    # Add dashboard title and description
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 1 - Company Profile')
    
    
    #create two tab columns
    col1, col2 = st.columns(2)

    #create dropbox for period selection
    period = col2.selectbox("Select Period", ('1mo','3mo','6mo','ytd','1y','3y','5y','max'))


    # create function for returning data from yfinance
    @st.cache
    def Get_stock_data(ticker,period):
        return yf.download(ticker,period=period,interval='1d')

    @st.cache
    def get_quote_table(ticker):
        return si.get_quote_table(ticker, dict_result=False)

    #if ticker is selected do the following
    if ticker != '-':
        #call function and store data in variable
        plot_data = Get_stock_data(ticker,period)
        #get quote table info
        quote_table = get_quote_table(ticker)
        #change value columns data type to pass to streamlit
        quote_table['value'] = quote_table['value'].astype(str)
        #create fig, ax objects respectively 
        fig, ax = plt.subplots(figsize=(15,10))
        #create a twin axes for the x axes
        ax_twin =ax.twinx()
        #create a bar chart to indicate stock volume
        ax_twin.bar(plot_data.index,plot_data['Volume'])
        #set y axes limit to five times the max in order to scale the bar chart appropriately
        ax_twin.set_ylim([0,plot_data['Volume'].max()*5])
        #plot close price data
        ax.plot(plot_data['Close'])
        #fill between x axes and highest point on the y axes
        ax.fill_between(plot_data.index,plot_data['Close'], color='green', alpha=0.5)
        ax.legend()
        col2.pyplot(fig)
        col1.dataframe(quote_table, height=1000)
        

       

#==============================================================================
# Tab 2
#==============================================================================

def tab2():
    
    # Add dashboard title and description
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 2 - Charts')
    
    # Add a check box
    show_data = st.checkbox("Display data")
    
    #create a selection boxes
    chart_type = st.sidebar.selectbox("Select Chart Type",('Line Plot', 'Candlestick Plot'))
    period = st.selectbox("Select Period", ('-','1mo','3mo','6mo','ytd','1y','3y','5y','max'))
    interval = st.selectbox("Select Interval", ('-','1d','1mo','3mo'))
     
     #some of the original code was sourced from class section 3's notebooks
    if ticker != '-' and period == '-' and interval != '-':
        stock_price = yf.download([ticker], start_date, end_date, interval=interval)
        if show_data:
            st.write('Stock price data')
            st.dataframe(stock_price)
    elif ticker != '-' and period == '-' and interval == '-':
        stock_price = yf.download([ticker], start_date, end_date, interval='1d')

    # if a period is selected us the following code to source the data
    if ticker != '-' and period != '-':
        stock_price = yf.download([ticker], period=period, interval='1d')
        if show_data:
            st.write('Stock price data')
            st.dataframe(stock_price)
        #if an interval is also selected use the following code to source the data
        if interval != '-':
            stock_price = yf.download([ticker], period=period, interval=interval)

    if chart_type == 'Line Plot':
        # If ticker selected create a line plot
        if ticker != '-':
            st.write('Adjusted close price')
            #create a fig, ax object 
            fig, ax = plt.subplots(figsize=(15, 5))
            #create a twin x axes 
            ax_twin =ax.twinx()
            #create a scaled bar chart to represent stock trading volume on the x axes
            ax_twin.bar(stock_price.index,stock_price['Volume'], color='red')
            ax_twin.set_ylim([0,stock_price['Volume'].max()*5])
            #rotate x axis tick lables
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.plot(stock_price['Adj Close'], label=ticker)
            #calculate simple moving average
            sma = stock_price['Adj Close'].rolling(window=50).mean()
            #slice sma series to get the values after the 50th index 
            ax.plot(sma.iloc[50:], label=f'{ticker} 50 Day SMA')
            ax.set_xlabel('Time')
            ax.set_ylabel('Adjust closing price (USD)')
            ax.legend()
            st.pyplot(fig)
    
    if chart_type == 'Candlestick Plot':
        if ticker != '-':
            # calculate 50 day simple moving average
            sma50 = stock_price['Close'].rolling(window=50).mean()

            # create fig with double y axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # add a candelstick plot to the fig
            fig.add_trace(go.Candlestick(x=stock_price.index, open=stock_price['Open'], high=stock_price['High'], low=stock_price['Low'], close=stock_price['Close'],
                             yaxis='y1', name='Candelstick'))
            #add plot for SMA 
            fig.add_trace(go.Scatter(x=stock_price.index, y=sma50.iloc[50:], name='5O Day Moving Average',
                        line=dict(color='yellow',width=2)))
            #add barchart for representing trading volume
            fig.add_trace(go.Bar(x=stock_price.index, y=stock_price['Volume'], yaxis='y2', name='Volume'))

            # Add figure title
            fig.update_layout(
            width=1100,
            height=600,
            title_text=f"{ticker}",
            yaxis_tickformat='M'
            )

            #configure layout
            fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            ))

            # Set x-axis label
            fig.update_xaxes(title_text="Date")

            # Set y-axes labels
            fig.update_yaxes(title_text="<b>primary</b> Close", secondary_y=False)
            fig.update_yaxes(title_text="<b>secondary</b> Volume", range=[0, 300000000], secondary_y=True)

            st.plotly_chart(fig)
            #much of this code was sourced from the link below:
            #https://stackoverflow.com/questions/64074854/how-to-enable-secondary-y-axis-in-plotly-get-better-visuals

#==============================================================================
# Tab 3
#==============================================================================

def tab3():

      # Add dashboard title and description
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 3 - Statistics')
    
    #create two tab columns
    col1, col2 = st.columns(2)

    #if ticker is select get relevant data
    if ticker != '-':
        #source data for stats
        stats = si.get_stats(ticker)
        #source valuation measures data
        valuation_stats = si.get_stats_valuation(ticker)
        #subset price history
        price_history = stats.iloc[0:7]

        col2.header('Trading Information')
        col2.write("Stock Price History")
        col2.dataframe(price_history, height=500)

        col2.write("Share Statistics")
        #subset share statistics
        share_statistics = stats.iloc[7:19]
        col2.dataframe(share_statistics, height=800)

        col2.write("Dividends & Splits")
        #subset dividends_split
        dividends_split = stats.iloc[19:29]
        col2.dataframe(dividends_split, height=800)

        col1.write("Valuation Measures")
        col1.dataframe(valuation_stats, height=200)

        col1.write("Fiscal Year")
        #subset fiscal year data
        fiscal_year = stats.iloc[29:31]
        col1.dataframe(fiscal_year, height=200)

        col1.write("Profitability")
        #subset profitability data
        profitability = stats.iloc[31:33]
        col1.dataframe(profitability, height=200)

        col1.write("Management Effectiveness")
        #subset management effectiveness data 
        manage_eff = stats.iloc[33:35]
        col1.dataframe(manage_eff, height=200)

        col1.write("Revenue")
        #subset revenue data
        i_s = stats.iloc[35]
        col1.dataframe(i_s, height=100)

    
    
    
#==============================================================================
# Tab 4
#==============================================================================

def tab4():
    
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 4 - Financials')
    
    #initiate period as False
    period= False

    #create buttons and conditional logic for changing function period parameter
    if st.button('Yearly'):
        period = True
    if st.button('Quarterly'):
        period = False

    @st.cache
    def get_balance_sheet(ticker, period):
        return si.get_balance_sheet(ticker, yearly=period)

    @st.cache
    def get_income_statement(ticker, period):
        return si.get_income_statement(ticker, yearly=period)

    @st.cache
    def get_cash_flow(ticker, period):
        return si.get_cash_flow(ticker, yearly=period)

    #if a ticker is selected 
    if ticker != '-':
        income_statement = get_income_statement(ticker,period)
        st.header('Income Statement')
        st.dataframe(income_statement,height=500,width=1000)
        balance_sheet = get_balance_sheet(ticker,period)
        st.header('Balance Sheet')
        st.dataframe(balance_sheet,height=400,width=1000)
        cash_flow = get_cash_flow(ticker,period)
        st.header('Cash Flow')
        st.dataframe(cash_flow,height=300, width=1000)





#==============================================================================
# Tab 5
#==============================================================================

def tab5():
    
     # Add dashboard title and description
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 5 Analysis')
    
    @st.cache
    def get_analysts_info(ticker):
        return si.get_analysts_info(ticker)

    if ticker != '-':
        analysis = get_analysts_info(ticker)
        #convert earnings estimate key value pair to dataframe
        earnings_estimate = pd.DataFrame.from_dict(analysis['Earnings Estimate'], orient='columns')
        st.header('Earnings Estimate')
        st.dataframe(earnings_estimate,height=300)
        #convert revenue estimate key value pair to dataframe
        revenue_estimate = pd.DataFrame.from_dict(analysis['Revenue Estimate'], orient='columns')
        st.header('Revenue Estimate')
        st.dataframe(revenue_estimate,height=300)
        #convert earnings history key value pair to dataframe
        earnings_history = pd.DataFrame.from_dict(analysis['Earnings History'], orient='columns')
        st.header('Earnings History')
        st.dataframe(earnings_history,height=300)
        #convert eps trend key value pair to dataframe
        eps_trend = pd.DataFrame.from_dict(analysis['EPS Trend'], orient='columns')
        st.header('EPS Trend')
        st.dataframe(eps_trend,height=300)
        #convert eps revisions key value pair to dataframe
        eps_revisions = pd.DataFrame.from_dict(analysis['EPS Revisions'], orient='columns')
        st.header('EPS Revision')
        st.dataframe(eps_revisions,height=300)
        #convert growth estimates key value pair to dataframe
        growth_estimates = pd.DataFrame.from_dict(analysis['Growth Estimates'], orient='columns')
        st.header('Growth Estimates')
        st.dataframe(growth_estimates,height=300)

#==============================================================================
# Tab 6
#==============================================================================

def tab6():
    # Add dashboard title and description
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 6 Simulation')

    @st.cache
    def get_stock_data(ticker,start_date,end_date):
        return si.get_data(ticker,start_date,end_date)

    simulations = st.selectbox("Select Number of Simulations", (200,500,1000))
    time_horizon = st.selectbox("Select Time Horizon (days)", (30,60,90))

    if ticker != '-':
        stock_data = get_stock_data(ticker,start_date,end_date)
        close_price = stock_data['close']
        daily_return = close_price.pct_change()
        daily_volatility = np.std(daily_return)
        # Setup the Monte Carlo simulation
        np.random.seed(123)
        # Run the simulation
        simulation_df = pd.DataFrame()
        # the following code was sourced from the class notebook IESEG_FP_Section3_Solution_v2.0
        for i in range(simulations):
    
            # The list to store the next stock price
            next_price = []
    
            # Create the next stock price
            last_price = close_price[-1]
    
            for j in range(time_horizon):
             # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price
    
            # Store the result of the simulation
            simulation_df[i] = next_price
            # Plot the simulation stock price in the future
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10, forward=True)

        plt.plot(simulation_df)
        plt.title('Monte Carlo simulation for AAPL stock price in next 200 days')
        plt.xlabel('Day')
        plt.ylabel('Price')

        plt.axhline(y=close_price[-1], color='red')
        plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')

        st.pyplot(fig)

        @st.cache
        def value_at_risk(df):
            # Price at 95% confidence interval
            future_price_95ci = np.percentile(df.iloc[-1:, :].values[0, ], 5)

            # Value at Risk
            VaR = stock_data['close'][-1] - future_price_95ci
            VaR_string = str(np.round(VaR, 2))
            return VaR_string
        
        var = value_at_risk(simulation_df)

        st.header(f"VaR at 95% confidence interval is: {var} USD")


#==============================================================================
# Tab 7
#==============================================================================

def tab7():
    st.title("Dimitri's Financial Dashboard")
    st.write("Data source: Yahoo Finance")
    st.header('Tab 7 Company Information')

    @st.cache
    def get_company_info(ticker):
        return si.get_company_info(ticker)

    @st.cache
    def get_earnings(ticker):
        return si.get_earnings_history(ticker)

    #create two tab columns
    col1, col2 = st.columns(2)

    if ticker != '-':
        #get company info data
        company_info = get_company_info(ticker)
        #convert value column to str to pass to streamlit
        company_info['Value'] = company_info['Value'].astype('str')
        col1.dataframe(company_info,height=500)
        #get earnings history data 
        earnings = get_earnings(ticker)
        #convert earnings history dictionary to dataframe
        earnings_dataframe = pd.DataFrame.from_dict(earnings,orient='columns')

        #subset estimated eps
        epsEstimate = earnings_dataframe['epsestimate']

        #subset acutal eps
        epsActual = earnings_dataframe['epsactual']

        
        fig, ax = plt.subplots(figsize=(15,10))

        #create scatter plot and plot estimated eps
        ax.scatter(earnings_dataframe.index,epsEstimate, color='red',label='EPS Estimate', alpha=0.5)
        #plot actual eps
        ax.scatter(earnings_dataframe.index, epsActual, color='blue',label='EPS Actual',alpha=0.5)
        ax.set_ylabel('Earnings per share')
        ax.set_xlabel('Quarters active')
        ax.legend()

        col2.pyplot(fig)




        
#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    
    # Add selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    global time_interval
    
    
    # Add select begin-end date
    global start_date, end_date
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())

    @st.cache
    def GetStockData(ticker, start_date, end_date):
        return si.get_data(ticker, start_date, end_date)


    if ticker != '-':
        stock_data = GetStockData(ticker, start_date, end_date)
    

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    if ticker != '-':
        csv = convert_df(stock_data)
        #create update and download button
        st.download_button(
        label="Update and Download Data as CSV",
        data=csv,
        file_name='stock_data.csv',
        mime='text/csv',
        )
    
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Company Profile', 'Charts','Statistics','Financials','Analysis', 'Monte Carlo Simulation','Other Analysis'])
    
    # Show the selected tab
    if select_tab == 'Company Profile':
        # Run tab 1
        tab1()
    elif select_tab == 'Charts':
        # Run tab 2
        tab2()
    elif select_tab == 'Statistics':
        tab3()
    elif select_tab == 'Financials':
        tab4()
    elif select_tab == 'Analysis':
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        tab6()
    elif select_tab == 'Other Analysis':
        tab7()
    
if __name__ == "__main__":
    run()
    
###############################################################################
# END
######################