import argparse

import code_train.A3C_Train as train

parser = argparse.ArgumentParser(description='Flatland with A3C')
parser.add_argument('--resume', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    train.start_train(args.resume)
