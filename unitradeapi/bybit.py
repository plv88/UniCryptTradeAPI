import requests
import sys
from datetime import datetime
import json
import traceback
import time
import websocket
from PlvLogger import Logger
import _thread
import threading

from queue import Queue


class Bybit_public:
    _main_url = r'https://api.bybit.com'

    def __init__(self, category):
        self.category = category
        self._logger = Logger('Bybit_public', type_log='w').logger
        self.headers = {}


    def __setattr__(self, key, value):
        if key == 'category' and value not in ['spot', 'der', 'linear', 'inverse', 'option']:
            self._logger.error(f'Неизвестный тип рынка {self.category}')
            raise TypeError(f"Неверный category {self.category}")
        if key == 'category' and value == 'der':
            value = 'linear'
        object.__setattr__(self, key, value)


    def _request_template(self, end_point, par=None, method='get'):
        work_link = self._main_url
        match method.lower():
            case 'get':
                req = requests.get(work_link + end_point, params=par)
            case 'post':
                req = requests.post(work_link + end_point, params=par)
            case 'delete':
                req = requests.delete(work_link + end_point, params=par)
            case _:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} Неизвестный метод {method}'
                print(mes_to_log)
                self._logger.error(mes_to_log)
                raise TypeError(f"Неверный method {mes_to_log}")
        if req.ok:
            return req.json()

    def get_instruments_info(self):
        """
        https://bybit-exchange.github.io/docs/v5/market/instrument
        """
        end_point = '/v5/market/instruments-info'
        par = {
            'category': self.category,
            'status': 'Trading'
        }
        return self._request_template(end_point=end_point, method='get', par=par)


    def get_symbols_in_trading(self):
        result = self.get_instruments_info().get('result')
        if result:
            return [el.get('symbol') for el in result.get('list') if el.get('status') == 'Trading']
        return None


class Bybit_websocket_public:
    _dict_urls = {
        'spot': 'wss://stream.bybit.com/v5/public/spot',
        'der': 'wss://stream.bybit.com/v5/public/linear',
        'der_auth': 'wss://stream.bybit.com/v5/private',
        'spot_auth': 'wss://stream.bybit.com/v5/private'
    }

    def __init__(self, trade_type, queue, topics):
        if trade_type not in self._dict_urls:
            raise ValueError(f"trade_type '{trade_type}' is not valid.")
        self.trade_type = trade_type
        self._logger = Logger('Bybit_websocket_public', type_log='w').logger
        self.queue = queue
        self.topics = topics
        self.websocket_app = websocket.WebSocketApp(
            url=self._dict_urls.get(trade_type),
            on_message=self.on_message,
            on_close=self.on_close,
            on_error=self.on_error,
            on_open=self.on_open,
        )
        self._is_running = True

    def __del__(self):
        self.websocket_app.close()

    def send_heartbeat(self, _ws):
        while self._is_running:  # Проверьте, что соединение должно быть открыто
            try:
                if _ws.sock and _ws.sock.connected:  # Проверьте, что сокет существует и соединение открыто
                    _ws.send(json.dumps({"req_id": "100001", "op": "ping"}))
                else:
                    print("WebSocket is not connected.")
                    break  # Выход из цикла, если соединение закрыто
                time.sleep(20)
            except websocket.WebSocketConnectionClosedException:
                print("WebSocket connection is closed, stopping heartbeat.")
                break

    def on_open(self, _wsapp):
        print("Connection opened")
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, args=(_wsapp,))
        self.heartbeat_thread.start()
        data = {
            "op": "subscribe",
            "args": self.topics
        }
        _wsapp.send(json.dumps(data))

    def on_close(self, _wsapp, close_status_code, close_msg):
        if close_status_code is not None and close_msg is not None:
            print(f"Close connection by server, status {close_status_code}, close message {close_msg}")

    def on_error(self, _wsapp, error):
        def_name = sys._getframe().f_code.co_name
        mes_to_log = f'{def_name} Error: {error}, traceback: {traceback.format_exc()}'
        self._logger.error(mes_to_log)
        print(mes_to_log)
        raise TypeError(mes_to_log)

    def stop(self):
        self._is_running = False  # Установите флаг в False, чтобы остановить пинг
        if self.websocket_app:
            self.websocket_app.close()
        if hasattr(self, 'heartbeat_thread'):
            self.heartbeat_thread.join()  # Дождитесь завершения потока пинга

    def on_message(self, _wsapp, message):
        parsed = json.loads(message)
        parsed['trade_type'] = self.trade_type
        # print(len(parsed), parsed)
        self.queue.put(parsed)



