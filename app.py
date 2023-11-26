from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
import time
#import tkinter as tk
#from tkinter import ttk
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

#Global Declarations
app = Flask(__name__)


# Initialize Binance API
client = Client("MY_API_KEY", "MY_API_SECRET")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_script", methods=["GET","POST"])
def run_script():
    try:
        # Retrieve the cryptocurrency name and symbol from the form
        cryptocurrency_name = request.form.get("cryptocurrency_name")
        cryptocurrency_symbol_pair = request.form.get("cryptocurrency_symbol_pair")

        #Pass information to tradingbot algorithim

         # Check if cryptocurrency_name is None or empty
        if not cryptocurrency_name:
            return render_template("result.html", script_output="Cryptocurrency name not provided.")
        elif request.form.get("long"):
             # Run trading bot and save script output
             script_output = runLongTradingBot(cryptocurrency_name, cryptocurrency_symbol_pair)
             # render output from  as html page
             return render_template("result.html", script_output=script_output)
        elif request.form.get("short"):
             script_output = runShortTradingBot(cryptocurrency_name, cryptocurrency_symbol_pair)
            # msg = "msg"
            # render output from  as html page
             return render_template("result.html", script_output=script_output,)  #msg = msg
    except Exception as e:
        return render_template({"error": f"Script execution failed: {str(e)}"})



def fetch_current_price(symbol):
    # Fetch current price of coin from Binance
    symbol_pair = symbol
    ticker = client.get_ticker(symbol=symbol_pair)
    return float(ticker["lastPrice"])


def fetch_historical_prices(symbol, interval, limit):
    # Fetch historical klines (candlestick) data from Binance
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    # Extract closing prices from klines
    prices = [float(kline[4]) for kline in klines]
    return prices


def calculate_ma_percentage_changes(short_ma, long_ma):
    percentage_change = ((short_ma - long_ma) / long_ma) * 100
    return percentage_change


def calculate_moving_averages(prices, short_window=10, long_window=50):
    if not isinstance(prices, (list, np.ndarray)):
        prices = [prices]  # Convert scalar to a list

    if len(prices) < long_window:
        return None, None

    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])
    return short_ma, long_ma


def calculate_rsi(prices, window=14):
    if not isinstance(prices, (list, np.ndarray)):
        prices = [prices]  # Convert scalar to a list

    if len(prices) < window:
        return None

    delta = np.diff(prices)
    gains = (delta[delta > 0]).mean()
    losses = (-delta[delta < 0]).mean()

    if gains == 0:
        return 0  # RSI is 0 when gains are 0

    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return rsi[-1] if isinstance(rsi, np.ndarray) else rsi


def swing_trade_strategy(
    current_price, short_ma, long_ma, rsi, rsi_overbought=70, rsi_oversold=30
):
    if short_ma > long_ma and rsi < rsi_oversold:
        return "buy"
    elif short_ma < long_ma or rsi > rsi_overbought:
        return "sell"
    else:
        return "hold"


def calculate_profit_and_loss(price):
    stop_loss_percentage = 0.02  # 2% stop-loss
    take_profit_percentage = 0.05  # 3% take-profit
    take_profit = price * (1 + take_profit_percentage)
    stop_loss = price * (1 - stop_loss_percentage)

    return take_profit, stop_loss


def calculate_trigger_price(current_price, short_ma, long_ma):
    ma_percentage_change = calculate_ma_percentage_changes(short_ma, long_ma)
    triggerPrice = current_price * (1 + ma_percentage_change / 100)
    return triggerPrice


def calculate_trigger_price_for_shorting(current_price, short_ma, long_ma):
    ma_percentage_change = calculate_ma_percentage_changes(short_ma, long_ma)
    triggerPrice = current_price * (
        1 - ma_percentage_change / 100
    )  # Adjusted for shorting
    return triggerPrice


def calculate_profit_and_loss_for_shorting(trigger_price):
    stop_loss_percentage = 0.02  # 2% stop-loss
    take_profit_percentage = 0.05  # 5% take-profit

    # For shorting, take-profit is below trigger price and stop-loss is above trigger price
    take_profit = trigger_price * (1 - take_profit_percentage)
    stop_loss = trigger_price * (1 + stop_loss_percentage)

    return take_profit, stop_loss


# def positionStatus(price):
#implement

# Trading logic loop


def runLongTradingBot(cryptoname,pair):
    global buy_output  # Declare trade_output as a global variable
    buy_output = []  # Initialize trade_output
    #thisdict = { "top" : "", "middle" : "","bottom" : "",}

    # To be removed and positionstatus
    position_held = False
    symbol_pair = pair
    name = cryptoname

    while True:
        try:
            # Get Ethereum price
            current_price = fetch_current_price(symbol_pair)
            "Trading bot is running.../n"
            buy_output.append(
                f"Cryptocurrency name: {name} and Symbol/Pair: {symbol_pair} "
            )
            print("Trading bot is running...\n")
            buy_output.append("Trading bot is running...\n")

            buy_output.append(f"Current {symbol_pair} price: {current_price}")

            # Fetch historical prices for calculating moving averages and RSI
            historical_prices = fetch_historical_prices(
                symbol_pair, Client.KLINE_INTERVAL_1DAY, 100
            )

            # Calculate moving averages
            moving_averages = calculate_moving_averages(historical_prices)
            if moving_averages:
                short_ma, long_ma = moving_averages
                rsi = calculate_rsi(historical_prices)
            else:
                short_ma, long_ma, rsi = None, None, None

            # Calculate RSI
            rsi = calculate_rsi(historical_prices)

            # Implement swing trade strategy
            signal = swing_trade_strategy(current_price, short_ma, long_ma, rsi)

            # Execute trade based on the signal
            if signal == "buy" and not position_held:
                buy_output.append(f"Signal: buy at this current price.\n")
                position_held = True  # Update position status
            elif signal == "sell" and position_held:
                buy_output.append(f"Signal: sell at this current price.\n")
                position_held = False  # Update position status
            else:
                buy_output.append(
                    f"No positions being held. Assessing {symbol_pair} price movements.\n"
                )
                #thisdict.update({"head": buy_output})

            rsi_oversold = 30  #  rsi_oversold threshold
            # Calculate the percentage change in moving averages
            percentage_change = calculate_ma_percentage_changes(short_ma, long_ma)

            # Calculate short_ma and determine the corresponding rsi
            triggerPrice = calculate_trigger_price(current_price, short_ma, long_ma)

            triggerPrice = round(triggerPrice, 5)
            corresponding_rsi = rsi_oversold

            # Print the price at which a buy trade would be triggere
            buy_output.append(f"Long")

            buy_output.append(f"Price to trigger a buy trade: {triggerPrice}")
            buy_output.append(
                f"Moving Average Percentage Change: {round(percentage_change,3)}"
            )

            # Print calculate and profit/loss at trigger price
            take_profit, stop_loss = calculate_profit_and_loss(triggerPrice)
            buy_output.append(f"Take Profit: {round(take_profit,5)} (+5%)")
            buy_output.append(f"Stop Loss: {round(stop_loss,5)} (-2%)\n")

            buy_output_str = "\n".join(buy_output)

            # Clear the trade output for the next iteration
            buy_output = []

            # Return the trade output for Flask to capture
            return buy_output_str  # Return the trade output for Flask to capture

        except Exception as e:
            print("An error occurred:", str(e))

def runShortTradingBot(cryptoname,pair):
    global sell_output  # Declare trade_output as a global variable
    sell_output = []  # Initialize trade_output

    # To be removed and positionstatus
    position_held = False
    symbol_pair = pair
    name = cryptoname

    while True:
        try:
            # Get Ethereum price
            current_price = fetch_current_price(symbol_pair)
            "Trading bot is running.../n"
            sell_output.append(
                f"Cryptocurrency name: {name} and Symbol/Pair: {symbol_pair} "
            )
            print("Trading bot is running...\n")
            sell_output.append("Trading bot is running...\n")

            sell_output.append(f"Current {symbol_pair} price: {current_price}")

            # Fetch historical prices for calculating moving averages and RSI
            historical_prices = fetch_historical_prices(
                symbol_pair, Client.KLINE_INTERVAL_1DAY, 100
            )

            # Calculate moving averages
            moving_averages = calculate_moving_averages(historical_prices)
            if moving_averages:
                short_ma, long_ma = moving_averages
                rsi = calculate_rsi(historical_prices)
            else:
                short_ma, long_ma, rsi = None, None, None

            # Calculate RSI
            rsi = calculate_rsi(historical_prices)

            # Implement swing trade strategy
            signal = swing_trade_strategy(current_price, short_ma, long_ma, rsi)

            # Execute trade based on the signal
            if signal == "buy" and not position_held:
                sell_output.append(f"Signal: buy at this current price.\n")
                position_held = True  # Update position status
            elif signal == "sell" and position_held:
                sell_output.append(f"Signal: sell at this current price.\n")
                position_held = False  # Update position status
            else:
                sell_output.append(
                    f"No positions being held. Assessing {symbol_pair} price movements.\n"
                )

            rsi_oversold = 30  #  rsi_oversold threshold
            # Calculate the percentage change in moving averages
            percentage_change = calculate_ma_percentage_changes(short_ma, long_ma)

            # Calculate short_ma and determine the corresponding rsi
            triggerPrice = calculate_trigger_price(current_price, short_ma, long_ma)

            triggerPrice = round(triggerPrice, 5)
            corresponding_rsi = rsi_oversold


            # Shorting
            triggerPrice = calculate_trigger_price_for_shorting(
                current_price, short_ma, long_ma
            )
            triggerPrice = round(triggerPrice, 5)
            take_profit, stop_loss = calculate_profit_and_loss_for_shorting(
                triggerPrice
            )

            sell_output.append(f"Short:")

            sell_output.append(f"Price to trigger a buy trade: {triggerPrice}")
            sell_output.append(
                f"Moving Average Percentage Change: {round(percentage_change,3)}"
            )
            sell_output.append(f"Take Profit: {round(take_profit,5)} (+5%)")
            sell_output.append(f"Stop Loss: {round(stop_loss,5)} (-2%)\n")

            sell_output_str = "\n".join(sell_output)

            # Clear the trade output for the next iteration
            sell_output = []

            # Return the trade output for Flask to capture
            return sell_output_str  # Return the trade output for Flask to capture

        except Exception as e:
            print("An error occurred:", str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
    app.run(debug=True)

