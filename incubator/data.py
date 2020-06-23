import pandas as pd
from pathlib import Path

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_unicode(data: pd.DataFrame) -> pd.DataFrame:
    "Replace the Unicode characters with their appropriate replacements."
    data['Utterance'] = (
        data.Utterance.apply(lambda s: s.replace('\x92', "'"))
            .apply(lambda s: s.replace('\x85', ". "))
            .apply(lambda s: s.replace('\x97', " "))
            .apply(lambda s: s.replace('\x91', ""))
            .apply(lambda s: s.replace('\x93', ""))
            .apply(lambda s: s.replace('\xa0', ""))
            .apply(lambda s: s.replace('\x94', ""))
    )
    return data
