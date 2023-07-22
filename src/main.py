import argparse

import torch

from settings import config
from utils.setup import create_stdout_handler

create_stdout_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline executor")
