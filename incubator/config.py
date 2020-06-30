import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
WANDB_KEY = os.getenv('WANDB_KEY')

