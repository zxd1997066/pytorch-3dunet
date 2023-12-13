import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DPredict')


def _get_output_file(dataset, suffix='_predictions'):
    return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, loader, output_file, config, **predictor_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    logger.info('Loading HDF5 datasets...')
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']
    batch_size = loaders_config.get('batch_size', 1)
    print("Batch Size: {}".format(batch_size))
    if config['channels_last']:
        try:
            model = model.to(memory_format=torch.channels_last_3d)
            print("[INFO] Use NHWC model.")
        except:
            print("[WARN] Model NHWC failed! Use normal model.")
    if config['compile']:
        model = torch.compile(model, backend=config['backend'], options={"freezing": True})
    if config['precision'] == "bfloat16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            for index, test_loader in enumerate(get_test_loaders(config)):
                logger.info(f"Processing '{test_loader.dataset.file_path}'...")
                output_file = _get_output_file(test_loader.dataset)
                predictor = _get_predictor(model, test_loader, output_file, config)
                # run the model prediction on the entire dataset and save to the 'output_file' H5
                predictor.predict()
                break
    elif config['precision'] == "float16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
            for index, test_loader in enumerate(get_test_loaders(config)):
                logger.info(f"Processing '{test_loader.dataset.file_path}'...")
                output_file = _get_output_file(test_loader.dataset)
                predictor = _get_predictor(model, test_loader, output_file, config)
                # run the model prediction on the entire dataset and save to the 'output_file' H5
                predictor.predict()
                break
    else:
        for index, test_loader in enumerate(get_test_loaders(config)):
            logger.info(f"Processing '{test_loader.dataset.file_path}'...")
            output_file = _get_output_file(test_loader.dataset)
            predictor = _get_predictor(model, test_loader, output_file, config)
            # run the model prediction on the entire dataset and save to the 'output_file' H5
            predictor.predict()
            break


if __name__ == '__main__':
    main()

