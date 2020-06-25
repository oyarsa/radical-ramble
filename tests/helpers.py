from io import StringIO
from pathlib import Path

import pandas as pd


test_data_str = """
Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
1,"Oh my God, hes lost it. Hes totally lost it.",Phoebe,sadness,negative,0,0,4,7,"00:20:57,256","00:21:00,049"
2,What?,Monica,surprise,negative,0,1,4,7,"00:21:01,927","00:21:03,261"
3,"Or! Or, we could go to the bank, close our accounts and cut them off at the source.",Ross,neutral,neutral,1,0,4,4,"00:12:24,660","00:12:30,915"
"""
test_tokens = [
    ['Oh', 'my', 'God', 'he', "'s", 'lost', 'it', 'He', "'s",
     'totally', 'lost', 'it'],
    ['What'],
    ['Or', 'Or', 'we', 'could', 'go', 'to', 'the', 'bank', 'close', 'our', 'accounts', 'and', 'cut', 'them', 'off', 'at', 'the', 'source'],
]

def read_test_data() -> pd.DataFrame:
    raw_data = StringIO(test_data_str)
    return pd.read_csv(raw_data)
