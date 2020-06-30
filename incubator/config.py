import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

defaults = {
    'glove_path': Path('./data/glove/glove.6B.50d.txt'),
    'glove_dim': 50,
}

