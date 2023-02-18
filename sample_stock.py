import argparse
import plotly.graph_objs as go
from plotly.offline import plot
import os
from rsi_divergence_finder import *
from timeframe import TimeFrame
import talib
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import smtplib
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))


Columns = ['ticker', 'Datetime', 'signal', 'Close', 'rsi', 'ema9', 'BBANDS_U', 'BBANDS_L', 'BBANDS_M']

Periods = {}
for i in range(1, 20, 1):
    key = f'{i}P'
    Periods[key] = i
    Columns.append(key)


class Email:
    def __init__(self):
        self.email = "arifmanasiya@gmail.com"
        self.password = "******"
        # establishing connection with gmail
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        self.server.ehlo()
        self.server.login(self.email, self.password)

    def send(self, html_message):
        # reading the spreadsheet
        email_list = pd.read_excel('/Users/arifman/dev/RSI-divergence-detector/email.xlsx')

        # getting the names and the emails
        names = email_list['NAME']
        emails = email_list['EMAIL']

        # iterate through the records
        for i in range(len(emails)):
            # for every record get the name and the email addresses
            name = names[i]
            email = emails[i]
            # sending the email
            self.server.sendmail(self.email, [email], html_message)

        # close the smtp server
        self.server.close()


real_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(real_path)


def plot_rsi_divergence(candles_df, divergences, pair, file_name):
    plot_file_name = os.path.join(os.getcwd(), '{}.html'.format(file_name))
    all_traces = list()

    all_traces.append(go.Scatter(
        x=candles_df['Date'].tolist(),
        y=candles_df['Close'].values.tolist(),
        mode='lines',
        name='Price'
    ))

    all_traces.append(go.Scatter(
        x=candles_df['Date'].tolist(),
        y=candles_df['rsi'].values.tolist(),
        mode='lines',
        name='RSI',
        xaxis='x2',
        yaxis='y2'
    ))

    for divergence in divergences:
        dtm_list = [divergence['start_dtm'], divergence['end_dtm']]
        rsi_list = [divergence['rsi_start'], divergence['rsi_end']]
        price_list = [divergence['price_start'], divergence['price_end']]

        color = 'rgb(0,0,255)' if 'bullish' in divergence['type'] else 'rgb(255,0,0)'

        all_traces.append(go.Scatter(
            x=dtm_list,
            y=rsi_list,
            mode='lines',
            xaxis='x2',
            yaxis='y2',
            line=dict(
                color=color,
                width=2)
        ))

        all_traces.append(go.Scatter(
            x=dtm_list,
            y=price_list,
            mode='lines',
            line=dict(
                color=color,
                width=2)
        ))

    layout = go.Layout(
        title='{} - RSI divergences'.format(pair),
        yaxis=dict(
            domain=[0.52, 1]
        ),
        yaxis2=dict(
            domain=[0, 0.5],
            anchor='x2'
        )
    )

    fig = dict(data=all_traces, layout=layout)
    plot(fig, filename=plot_file_name)


def plot_returns(signal_df, per=1):
    return_series_col = f"{per}P"
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    series = signal_df[return_series_col]
    series.plot.hist(bins=60)
    ax1.set_xlabel(f"{per} Period returns %")
    ax1.set_ylabel("# of Signals")
    ax1.set_title(f"{per} Period Returns Data")
    ax1.text(-0.35, 200, "Extreme Low\nreturns")
    ax1.text(0.25, 200, "Extreme High\nreturns")
    plt.savefig(f'{per}_period_returns.png')


def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope


def get_data(symbol, interval, period, _args):
    data = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        ignore_tz=False,
        # group_by='ticker',
        auto_adjust=True,
        repair=False,
        prepost=False,
        threads=True,
        proxy=None,
        progress=False
    )

    if not data.shape[0]:
        return None

    data['Close'] = pd.to_numeric(data['Close'])
    data['ohlc'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4.0
    ref_col = data[_args.reference]
    data['BBANDS_U'] = talib.BBANDS(ref_col, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)[0]
    data['BBANDS_M'] = talib.BBANDS(ref_col, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)[1]
    data['BBANDS_L'] = talib.BBANDS(ref_col, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)[2]
    data['ema9'] = talib.EMA(ref_col, 9)
    data['ema_gap'] = (ref_col - data['ema9']).abs()

    # set min_periods=2 to allow subsets less than 60.
    # use [4::5] to select the results you need.
    data['price_slope'] = talib.LINEARREG_ANGLE(normalize(ref_col).values, 14)
    data['volume_slope'] = talib.LINEARREG_ANGLE(normalize(pd.to_numeric(data['Volume'])), 14)
    data['rank'] = data['Volume']
    data[RSI_COLUMN] = talib.RSI(ref_col, 14)
    data.dropna()

    find_extrema(data, 3, _args.reference)
    data = data.reset_index()
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    data.stack(level=0).rename_axis(['Datetime', 'Ticker']).reset_index(level=1)
    latest = data.tail(1)
    data.dropna()  # remove all non peaks from the data
    data = data[data['min'].notna() | data['max'].notna()].copy()
    ref_col = data[_args.reference]  # Reset ref_col after drop na
    data['ema'] = latest['ema9'].max()
    data['ugap'] = np.where((data['max'].notna()) & ((data['BBANDS_U'] - ref_col) > 0), (data['BBANDS_U'] - ref_col), np.nan)
    data['lgap'] = np.where((data['min'].notna()) & ((ref_col - data['BBANDS_L']) > 0), (ref_col - data['BBANDS_L']), np.nan)
    data['CurrentClose'] = latest['Close'].max()
    data['CurrentOpen'] = latest['Open'].max()
    data['ticker'] = symbol
    data['signalTime'] = latest['Datetime'].max()
    time_diff = data['signalTime'] - data['Datetime']
    data['signalAgeInMin'] = time_diff.astype('timedelta64[m]')
    return data


def normalize(col):
    max_value = col.max()
    min_value = col.min()
    return (col - min_value) / (max_value - min_value)


def find_extrema(df, n, reference):
    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df[reference].values, np.less_equal, order=n)[0]][reference]
    df['max'] = df.iloc[argrelextrema(df[reference].values, np.greater_equal, order=n)[0]][reference]


def get_bears_and_bulls(frame, _args):
    reference_label = _args.reference
    bears = frame[frame['max'].notna()].copy()
    bears['pPeakTime'] = bears.Datetime.shift(1)
    bears['p2PeakTime'] = bears.Datetime.shift(2)
    bears['p3PeakTime'] = bears.Datetime.shift(3)

    bulls = frame[frame['min'].notna()].copy()
    bulls['pPeakTime'] = bulls.Datetime.shift(1)
    bulls['p2PeakTime'] = bulls.Datetime.shift(2)
    bulls['p3PeakTime'] = bulls.Datetime.shift(3)

    ref_bear = bears[reference_label]
    ref_bull = bulls[reference_label]

    bulls_result = pd.DataFrame()
    bears_result = pd.DataFrame()
    if 'RSI' in _args.factors:
        rsi_bears = bears.query(
            f"("
            f"( @ref_bear > @ref_bear.shift(1) ) "  # Peak price increases
            f"& ( rsi <= rsi.shift(1) ) "  # Peak RSI decreases
            f"& ( volume_slope >= volume_slope.median() ) "  # Decreasing volume
            f") "
        ).copy()
        rsi_bears['signal'] = "rsi_bear"
        bears_result = pd.concat([bears_result, rsi_bears])

        rsi_bulls = bulls.query(
            f"("
            f"( @ref_bull < @ref_bull.shift(1) ) "
            f"& ( rsi >= rsi.shift(1) ) "
            f"& ( volume_slope >= volume_slope.median() ) "  # Decreasing volume
            f")"
        ).copy()
        rsi_bulls['signal'] = "rsi_bull"
        bulls_result = pd.concat([rsi_bulls, bulls_result])

    if 'HIDDEN' in _args.factors:
        rsi_hidden_bears = bears.query(
            f"("
            f"( @ref_bear < @ref_bear.shift(1) ) "  # Peak price decreases
            f"& ( rsi > rsi.shift(1) ) "  # Peak RSI increases
            f") "
        ).copy()
        rsi_hidden_bears['signal'] = "rsi_hidden_bear"
        bears_result = pd.concat([bears_result, rsi_hidden_bears])

        rsi_hidden_bulls = bulls.query(
            f"("
            f"( @ref_bull > @ref_bull.shift(1) ) "
            f"& ( rsi < rsi.shift(1) ) "
            f")"
        ).copy()
        rsi_hidden_bulls['signal'] = "rsi_hidden_bulls"
        bulls_result = pd.concat([rsi_hidden_bulls, bulls_result])

    if 'BOLL' in _args.factors:
        stat_bears = bears.query(
            f"( @ref_bear >= @ref_bear.shift(1) ) "  # Closed at or above prev peak's high 
            f"& ( BBANDS_U.shift(1) < @ref_bear.shift(1) ) "  # Prev High over UB
            f"& ( @ref_bear <= BBANDS_U ) "  # Closed below UB
            f"& ( ugap < ugap.quantile(0.4) ) "  # Peaks gap from upper band when within.
            f"& ( volume_slope >= volume_slope.median() ) "  # Decreasing volume
        ).copy()
        stat_bears['signal'] = "stat_bear"
        bears_result = pd.concat([stat_bears, bears_result])

        stat_bulls = bulls.query(
            f"( @ref_bull <= @ref_bull.shift(1) ) "
            f"& ( BBANDS_L.shift(1) > @ref_bull.shift(1) ) "
            f"& ( @ref_bull >= BBANDS_L ) "
            f"& ( lgap < lgap.quantile(0.4) ) "  # Trough gap from lower band when within.
            f"& ( volume_slope >= volume_slope.median() ) "  # Decreasing volume
        ).copy()
        stat_bulls['signal'] = "stat_bull"
        bulls_result = pd.concat([stat_bulls, bulls_result])

    if "EMA" in _args.factors:
        ema_bears = bears.query(
            # close below EMA
            f"( @ref_bear <= ema ) "
            # Green bar did not close over EMA (rejection from EMA)
            f"& ( Close >= Open ) "
            # Previous bar (peak) was Green
            f"& ( Close.shift(1) >= Open.shift(1) ) "
            # Previous bar (peak) closed above EMA
            f"& ( @ref_bear.shift(1) >= ema.shift(1) ) "
            # minimal gap from the EMA
            f"& ( ema_gap < ema_gap.quantile(0.1) ) "
            # Gap of the prev peak is also minimal
            f"& ( ugap.shift(1) < ugap.quantile(0.2) ) "
        ).copy()
        ema_bears['signal'] = "ema_bear"
        bears_result = pd.concat([ema_bears, bears_result])

        ema_bulls = bulls.query(
            # Closes above EMA
            f"( @ref_bull >= ema ) "
            # Current bar is Red
            f"& ( Close <= Open ) "
            # Previous bar (peak) is Red
            f"& ( Close.shift(1) < Open.shift(1) ) "  
            # Previous bar(peak) closed below EMA
            f"& ( @ref_bull.shift(1) <= ema.shift(1) ) "  
            # Gap between current close and EMA is minimal
            f"& ( ema_gap < ema_gap.quantile(0.1) ) "
            f"& ( lgap < lgap.quantile(0.2) ) "
        ).copy()
        ema_bulls['signal'] = "ema_bull"
        bulls_result = pd.concat([ema_bulls, bulls_result])

    bulls_result.drop_duplicates()
    bears_result.drop_duplicates()

    result = pd.concat([bulls_result, bears_result])
    result.sort_values(
        by=['Datetime'],
        inplace=True,
        ascending=False
    )
    return result


def get_signals(data_frame, _args):
    res = get_bears_and_bulls(data_frame, _args)
    timeframe = options[args.timeframe]
    max_age = _args.maxAge * timeframe.value[0] or timeframe.value[0]
    res = res[res['signalAgeInMin'] <= max_age].copy()
    return res


options = {
    "1m": TimeFrame.ONE_MIN,
    "5m": TimeFrame.FIVE_MIN,
    "15m": TimeFrame.FIFTEEN_MIN,
    "1h": TimeFrame.ONE_HOUR,
    "1d": TimeFrame.ONE_DAY,
    "1w": TimeFrame.ONE_WEEK
}


def display(frame, _args):
    if frame.shape[0]:
        frame.sort_values(
            by=['Datetime', 'rank', 'Volume', 'ticker'],
            inplace=True,
            ascending=[False, False, True, True])
        file_base = f'./output/{options[_args.timeframe].value[1]}_{_args.reference}_{_args.exchange}_signal'
        frame.to_csv(open(f'{file_base}.csv', 'w'))
        # TODO: Telegram or Email
        print(frame[['ticker', 'Datetime', 'signal', 'signalAgeInMin']])


def init_args():
    parser = argparse.ArgumentParser(description='Process Timeframe')
    parser.add_argument("-t", "--timeframe", help="Enter timeframe 1m, 5m, 15m, 1h, 1d, 1w", required=True)
    parser.add_argument("-s", "--symbol", help="Enter Ticker Symbol", required=False)
    parser.add_argument("-e", "--exchange", help="Enter Exchange (NSE)", required=False)
    parser.add_argument("-a", "--maxAge", help="Enter max signal age in minutes", required=False, type=int, default=1)
    parser.add_argument("-i", "--input", help="Enter input symbol file", required=False, type=str)
    parser.add_argument("-r", "--reference", help="Enter data reference (Open, Close, ohlc)",
                        required=False, type=str, default="Close")
    parser.add_argument("-f", "--factors", help="Enter factors like rsi, stat, ema",
                        required=False, type=str, default=['RSI', 'BOLL', 'EMA'], action=SplitArgs)
    return parser


def get_ticker_strings(_args):
    if _args.symbol is not None:
        ticker_strings = [_args.symbol]
        return ticker_strings
    elif _args.input is not None:
        symbol_file = _args.input
        if os.path.exists(symbol_file):
            ticker_strings = pd.read_csv(symbol_file)['ticker'].to_list()
            if len(ticker_strings):
                print(f"Using {len(ticker_strings)} symbols from {symbol_file} for ticker list.")
                return ticker_strings
    elif _args.exchange is not None:
        symbol_file = f'./input/{_args.exchange}_symbols.csv'
        if os.path.exists(symbol_file):
            ticker_strings = pd.read_csv(symbol_file)['Symbol'].to_list()
            if len(ticker_strings):
                print(f"Using {len(ticker_strings)} symbols from {symbol_file} for ticker list.")
                return ticker_strings

    ticker_lst = pd.read_csv('./input/options_symbol.csv')
    ticker_strings = ticker_lst['Symbol'].to_list()
    ticker_strings = [s for s in ticker_strings if "." not in s]
    if _args.exchange == "NSE":
        tickers = pd.read_html(
            'https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[
            0]
        tickers = tickers.SYMBOL.to_list()
        for count in range(len(tickers)):
            tickers[count] = tickers[count] + ".NS"
            ticker_strings = tickers
    elif _args.exchange == "BEAR":
        ticker_strings = pd.read_csv("./input/1d_bear_signal.csv")['ticker']
    elif _args.exchange == "BULL":
        ticker_strings = pd.read_csv("./input/1d_bull_signal.csv")['ticker']
    elif _args.exchange == "SPX":
        df1 = pd.DataFrame(si.tickers_sp500())
        ticker_strings = ticker_strings + ([symbol for symbol in df1[0].values.tolist() if len(symbol) <= 4 and "." not in symbol])
    elif _args.exchange == "NDAQ":
        df1 = pd.DataFrame(si.tickers_nasdaq())
        ticker_strings = [symbol for symbol in df1[0].values.tolist() if len(symbol) <= 4]
    elif _args.exchange == "DOW":
        df1 = pd.DataFrame(si.tickers_dow())
        ticker_strings = [symbol for symbol in df1[0].values.tolist() if len(symbol) <= 4]
    else:
        df1 = pd.DataFrame(si.tickers_other())
        ticker_strings = [symbol for symbol in df1[0].values.tolist() if len(symbol) <= 4]

    return list(set(ticker_strings))


if __name__ == '__main__':

    par = init_args()
    args = par.parse_args()
    print(f"Processing for {args.timeframe} candles with reference to {args.reference}...")
    print(f"Evaluating factors in {args.factors}")
    time_frame = options[args.timeframe]
    all_signals = pd.DataFrame()
    processed_symbols = pd.DataFrame()
    ticker_list = get_ticker_strings(args)
    if not ticker_list:
        print("Please provide ticker(s)")
        exit(1)

    for ticker in ticker_list:
        dataDf = get_data(ticker, time_frame.value[1], time_frame.value[2], args)
        if dataDf is None:
            continue
        processed_symbols = pd.concat([processed_symbols, dataDf.head(1)])
        signals = get_signals(dataDf, args)
        if signals.shape[0] and args.symbol is not None:
            print(signals.tail(1).T)
        all_signals = pd.concat([signals, all_signals])

    display(all_signals, args)
    # if processed_symbols.shape[0]:
    #     processed_symbols.rename(columns={'ticker': 'Symbol'}, inplace=True)
    #     processed_symbols['Symbol'].to_csv(open(f'./input/{args.exchange or "default"}_symbols.csv', 'w'))
