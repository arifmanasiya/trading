import argparse

import plotly.graph_objs as go
from plotly.offline import plot
import os
import requests
from rsi_divergence_finder import *
from timeframe import TimeFrame
import talib


def init_args():
    parser = argparse.ArgumentParser(description='Process Timeframe')
    parser.add_argument("-f", "--file1", help="file name 1", required=False, default="1d_Close_None_signal.csv")
    parser.add_argument("-g", "--file2", help="file name 2", required=False, default="1d_ohlc_None_signal.csv")
    return parser


fields = ["ticker", "signal", "signalAgeInMin", "rsi"]

if __name__ == '__main__':
    par = init_args()
    args = par.parse_args()
    signal_file_1 = args.file1
    signal_file_2 = args.file2
    if os.path.exists(signal_file_1) and os.path.exists(signal_file_2):
        signals_1 = pd.read_csv(signal_file_1)[fields]
        signals_2 = pd.read_csv(signal_file_2)[fields]
        bears1 = signals_1.query('signal.str.contains("bear")').drop_duplicates()
        bulls1 = signals_1.query('signal.str.contains("bull")').drop_duplicates()
        bears2 = signals_2.query('signal.str.contains("bear")').drop_duplicates()
        bulls2 = signals_2.query('signal.str.contains("bull")').drop_duplicates()
        final_bears = pd.merge(bears1, bears2, how='inner', on=['ticker']).drop_duplicates()
        final_bears.sort_values(
            by=['signalAgeInMin_x', 'signalAgeInMin_y', 'rsi_y'],
            inplace=True,
            ascending=[True, True, False]
        )
        final_bulls = pd.merge(bulls1, bulls2, how='inner', on=['ticker']).drop_duplicates()
        final_bulls.sort_values(
            by=['signalAgeInMin_x', 'signalAgeInMin_y', 'rsi_y'],
            inplace=True,
            ascending=[True, True, True]
        )

        timeframe1 = signal_file_1.split('/')[1].split('_')[0]
        timeframe2 = signal_file_2.split('/')[1].split('_')[0]
        output = f"{timeframe1}_merged_{timeframe2}"
        final_bulls.to_csv(open(f'output/bulls_{output}.csv', 'w'))
        final_bears.to_csv(open(f'output/bears_{output}.csv', 'w'))
        print(final_bears)
        print(final_bulls)

