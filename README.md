# Cryptocurrency API Clients for Binance and Bybit

This project provides a comprehensive library for interacting with the APIs of the Binance and Bybit cryptocurrency exchanges. It's designed for developers and analysts who need to access market data, trading information, and statistical insights in real-time.

## Features

- **Versatile Market Data Access**: Retrieve real-time market data including exchange information, order books, recent trades, and more.
- **Support for Multiple Market Types**: Compatible with different market types such as spot, derivatives, futures, and options.
- **Advanced Trading Operations**: Facilitates advanced trading operations like historical trades lookup, aggregated trades lists, and K-line (candlestick data) queries.
- **Websocket Support**: Integration with Binance and Bybit Websockets for real-time data streaming.
- **Error Handling and Logging**: Robust error handling and logging capabilities for reliable application development.

## Installation

To install the library, use the following pip command:

```bash
pip install UniCryptTradeAPI
```

```commandline
from UniCryptTradeAPI import SyncBinancePublic, SyncBybitPublic

# Initialize clients
binance_client = SyncBinancePublic(market_type='spot')
bybit_client = SyncBybitPublic(category='spot')

# Example usage
exchange_info = binance_client.get_exchange_information()
order_book = bybit_client.get_order_book(symbol='BTCUSD')
```

## Main methods from Bybit
SyncBybitPublic:
- instrument
- symbols_in_trading
- tickers

SyncBybitPrivate:
- wallet-balance
- transaction-log

AsyncBybitPrivate:
- wallet-balance
- transaction-log

SyncBybitWebsocketPublic:
- base class

    





