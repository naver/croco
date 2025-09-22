"""
Script to load a CroCo stereo/flow model, build its downstream head, 
and export it to ONNX format for deployment.
"""

import argparse
import os
from pathlib import Path
import logging

import torch
import onnx

from models.croco_downstream import CroCoDownstreamBinocular
from models.head_downstream import PixelwiseTaskWithDPT
from stereoflow.criterion import *


def get_args_parser():
    """
    Parse command-line arguments for model evaluation/export.
    """
    parser = argparse.ArgumentParser(
        'Export CroCo models (stereo/flow) to ONNX format',
        add_help=False
    )
    parser.add_argument(
        '--model', required=True, type=str,
        help='Path to the PyTorch model checkpoint to evaluate/export'
    )
    return parser


def _load_model_and_criterion(model_path, device):
    """
    Load a pretrained CroCo model checkpoint and reconstruct the model.

    Args:
        model_path (str): Path to the saved checkpoint file.
        device (torch.device): Target device to map the model to.

    Returns:
        model (torch.nn.Module): Loaded CroCo model in eval mode.
    """
    print('Loading model from', model_path)
    assert os.path.isfile(model_path), f"Checkpoint not found: {model_path}"

    # Load checkpoint (full state, not weights-only)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt['args']

    # Determine number of output channels depending on task
    task = ckpt_args.task
    num_channels = {'stereo': 1, 'flow': 2}[task]
    with_conf = eval(ckpt_args.criterion).with_conf
    if with_conf:
        num_channels += 1

    print('Head: PixelwiseTaskWithDPT()')
    head = PixelwiseTaskWithDPT()
    head.num_channels = num_channels

    print('CroCo args:', ckpt_args.croco_args)
    model = CroCoDownstreamBinocular(head, **ckpt_args.croco_args)
    model.eval()
    model = model.to(device)

    return model


def export_onnx(
    model,
    inputs,
    weights,
    input_names=None,
    opset=18,
    dynamic=False,
    simplify=True,
    prefix='ONNX:'
):
    """
    Export a PyTorch CroCo model to ONNX format.

    Args:
        model (torch.nn.Module): Model to export.
        inputs (tuple): Example input tensors.
        weights (str): Path (checkpoint) to derive ONNX filename.
        input_names (list[str]): Names for ONNX input nodes.
        opset (int): ONNX opset version.
        dynamic (bool): Enable dynamic axes (batch/height/width).
        simplify (bool): Run ONNX simplifier to optimize model.
        prefix (str): Logging prefix.

    Returns:
        f (Path): Path to the saved ONNX file.
        onnx_model (onnx.ModelProto): Simplified or raw ONNX model.
    """
    logger = logging.getLogger('torch.onnx')
    logger.info(f'{prefix} starting export with onnx {onnx.__version__}...')

    f = Path(weights).with_suffix('.onnx')

    # Define I/O node names
    output_names = ['disparity']

    # Dynamic shape configuration
    if dynamic:
        dynamic = {
            'left_img': {0: 'batch', 2: 'height', 3: 'width'},
            'right_img': {0: 'batch', 2: 'height', 3: 'width'}
        }

    # Export with torch.onnx
    torch.onnx.export(
        model,
        inputs,
        f,
        verbose=True,
        report=True,
        opset_version=opset,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None,
        dynamo=False
    )

    # Check and reload ONNX model
    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, f)

    # Simplify with onnx-simplifier if enabled
    if simplify:
        try:
            import onnxsim
            logger.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_opt, check = onnxsim.simplify(model_onnx)
            logger.info("Simplification diff info:")
            logger.info(onnxsim.model_info.print_simplifying_info(model_onnx, model_opt))
            assert check, 'Simplification check failed'
            onnx.save(model_opt, f)
            return f, model_opt
        except Exception as e:
            logger.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx


def main(args):
    """
    Main entry point:
      1. Load model checkpoint.
      2. Export it to ONNX using dummy input.
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = _load_model_and_criterion(args.model, device=device)

    # Dummy stereo input (batch=1, 3x352x704 RGB images for left & right views)
    x = torch.randn(1, 3, 352, 704).to(device)
    export_onnx(model, (x, x), f"{args.model}.onnx", input_names=['img1', 'img2'])


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)