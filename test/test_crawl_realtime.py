import os
import time

from FiinQuantX import FiinSession, RealTimeData
from dotenv import load_dotenv

load_dotenv(dotenv_path='../config/.env')

USERNAME = os.getenv("FIINQUANT_USERNAME")
PASSWORD = os.getenv("FIINQUANT_PASSWORD")

client = FiinSession(username=USERNAME, password=PASSWORD).login()

tickers = ['HPG','VNINDEX','VN30F1M']

def onTickerEvent(data: RealTimeData):
    print('----------------')
    print(data.to_dataFrame())
    # data.to_dataFrame().to_csv('callback.csv', mode='a', header=True)
Events = client.Trading_Data_Stream(tickers=tickers, callback = onTickerEvent)
Events.start()

try:
    while not Events._stop:
        time.sleep(1)
except KeyboardInterrupt:
    print("KeyboardInterrupt received, stopping...")
    Events.stop()