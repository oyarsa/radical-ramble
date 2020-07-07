"""Configuration from environment variables and default constants."""
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

defaults = {
    'glove_path': Path('./data/glove/glove.6B.50d.txt'),
    'glove_dim': 50,
    'train_data': Path('./data/train/metadata.csv'),
    'eval_data': Path('./data/test/metadata.csv'),
}
