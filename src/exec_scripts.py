import argparse
import json

import delete_checkpoints
import evaluate_predictions
import generate_submission_file
from get_loggers import get_logger
import map_multiclass_to_binary_preds
import predict
import ensemble_predict
import train


RUN_LOGGER = get_logger('run')


def main(args: argparse.Namespace) -> None:
    with open(args.path_config) as fin:
        config = json.load(fin)
    
    # # training
    # for key in config:
    #     if key.startswith('train'):
    #         RUN_LOGGER.info('Start training.')
    #         train.main(argparse.Namespace(**config[key]))
    #         RUN_LOGGER.info('Finished training.')
    
    # # deleting checkpoints
    # for key in config:
    #     if key.startswith('delete_checkpoints'):
    #         RUN_LOGGER.info('Delete checkpoints.')
    #         delete_checkpoints.main(argparse.Namespace(**config[key]))
    #         RUN_LOGGER.info('Finished deleting checkpoints.')
    
    # # predicting
    # for key in config:
    #     if key.startswith('predict'):
    #         # predict on both devsets
    #         RUN_LOGGER.info('Start prediction on internal dev set.')
    #         predict.main(argparse.Namespace(**config[key]))
    #         RUN_LOGGER.info('Finished prediction on internal dev set.')
    
    # ensemble predicting
    for key in config:
        if key.startswith('ensemble_predict'):
            # predict on both devsets
            RUN_LOGGER.info('Start ensemble prediction.')
            ensemble_predict.main(argparse.Namespace(**config[key]))
            RUN_LOGGER.info('Finished ensemble prediction.')
    
    # label mapping
    for key in config:
        if key.startswith('map_multiclass_to_binary_preds'):
            RUN_LOGGER.info('Start mapping onto binary labels.')
            map_multiclass_to_binary_preds.main(argparse.Namespace(**config[key]))
            RUN_LOGGER.info('Finished mapping onto binary labels.')

    # evaluation
    for key in config:
        if key.startswith('evaluate_predictions'):
            RUN_LOGGER.info('Evaluate predictions on internal dev set.')
            evaluate_predictions.main(argparse.Namespace(**config[key]))
            RUN_LOGGER.info('Finished evaluation of predictions on internal dev set.')
    
    # generation of submission files
    for key in config:
        if key.startswith('generate_submission_file'):
            RUN_LOGGER.info('Generate dev-set submission file.')
            generate_submission_file.main(argparse.Namespace(**config[key]))
            RUN_LOGGER.info('Finished generating dev-set submission file.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_config', help='Path to a json-file containing the following top-level keys: '
        '"train", "delete_checkpoints", "predict", "evaluate_predictions", "generate_submission_files". Each key has' 
        'a value a dictionary containing all arguments needed for the corresponding script. Optinonally the file can' 
        'also contain the arguments for multi-class to cinary label conversions.')
    cmd_args = parser.parse_args()
    main(cmd_args)
