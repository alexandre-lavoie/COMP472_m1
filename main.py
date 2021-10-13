from mini1 import *
import argparse
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COMP472 Mini-Project 1")
    parser.add_argument('task', choices=['bbc', 'drug'], help="Task to run")
    args = parser.parse_args()

    data_path="./data"
    if not os.path.exists(data_path): os.makedirs(data_path, exist_ok=True)
    
    result_path="./results"
    if not os.path.exists(result_path): os.makedirs(result_path, exist_ok=True)

    if args.task == 'bbc':
        print("BBC Training\n")

        bbc_main(
            data_path=data_path,
            result_path=result_path
        )
    elif args.task == 'drug':
        print("Drug Training\n")

        drug_main(
            data_path=data_path,
            result_path=result_path
        )
