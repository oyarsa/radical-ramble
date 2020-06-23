from pathlib import Path

from incubator import data

def test_clean_unicode() -> None:
    data_path = Path('./data/dev/metadata.csv')
    dev = data.load_data(data_path)
    dev = data.clean_unicode(dev)

    nonascii = dev[dev.Utterance.apply(lambda s: not s.isascii())]
    assert len(nonascii) == 0
