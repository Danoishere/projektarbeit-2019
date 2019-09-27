import code_play.A3C_Play as play
import code_train.A3C_Train as train
import argparse

parser = argparse.ArgumentParser(description='Flatland with A3C')
parser.add_argument('--train', action='store_true')
parser.add_argument('--play', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.play:
        play.start_play()
    elif args.train:
        train.start_train()