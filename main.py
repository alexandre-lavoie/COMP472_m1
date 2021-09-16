from mini1 import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COMP472 Mini-Project 1")
    parser.add_argument('task', choices=['bbc', 'drug'], help="Task to run")
    args = parser.parse_args()

    if args.task == 'bbc': bbc_main()
    elif args.task == 'drug': drug_main()
