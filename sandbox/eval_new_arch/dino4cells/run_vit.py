import logging
import os
import sys


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")


from sandbox.eval_new_arch.dino4cells.main_vit import train_vit


if __name__ == "__main__":    
    print("Calling the training func...")
    config_data_path = sys.argv[1]
    config_path = sys.argv[2]
    try:
        train_vit(config_path, config_data_path)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")

# ./bash_commands/run_py.sh ./sandbox/eval_new_arch/dino4cells/run_vit -g -m 40000 -b 44  -a ./src/datasets/configs/training_data_config/B78NoDSTrainDatasetAugInPlaceConfig ./sandbox/eval_new_arch/dino4cells/configs/config_naive_vit/base -j training_NOVA_vit -q gsla_high_gpu