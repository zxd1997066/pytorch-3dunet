import argparse

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--ipex', action='store_true', help='Use intel pytorch extension.')
    parser.add_argument('--jit', action='store_true', help='enable jit optimization in intel pytorch extension.')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size.')
    parser.add_argument('--num_iter', type=int, default=10, help='num of warmup, default is 10.')
    parser.add_argument('--num_warmup', type=int, default=0, help='num of warmup, default is 0.')
    parser.add_argument('--precision', type=str, default='float32', help='data type precision, default is float32.')
    parser.add_argument('--channels_last', type=int, default=1, help='Use NHWC.')
    parser.add_argument('--arch', type=str, default=None, help='model name.')
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")

    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    print(config)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    config['loaders']['batch_size'] = args.batch_size
    config['ipex'] = args.ipex
    config['num_iter'] = args.num_iter
    config['num_warmup'] = args.num_warmup
    config['jit'] = args.jit
    config['precision'] = args.precision
    config['channels_last'] = args.channels_last
    config['profile'] = args.profile
    config['compile'] = args.compile
    config['backend'] = args.backend
    config['triton_cpu'] = args.triton_cpu
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
