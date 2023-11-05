import logging
import onnx
import numpy as np
import onnx_graphsurgeon as gs
from pathlib import Path


_DEFAULT_LOGGER = logging.getLogger(__name__)


def convert_onnx_file(input_filepath, output_filepath, *, logger=None):
    """convert QDQ ONNX file from scale=0 to 1

    Parameters
    ----------
    input_filepath : str or Path
        input QDQ ONNX filepath
    output_filepath : str or Path
        output QDQ ONNX filepath
        if parent directory does not exist, make directory
    logger : logging.Logger, default=None
        a logger

    See Also
    --------
    convert_onnx_model : see this function for detailed convert steps
    """
    if not logger:
        logger = _DEFAULT_LOGGER

    logger.info(f'loading {input_filepath}...')
    input_model = onnx.load(str(input_filepath))

    output_model = convert_onnx_model(input_model, logger=logger)

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, str(output_filepath))
    logger.info(f'save to {output_filepath} done')


def convert_onnx_model(onnx_model, *, logger=None):
    """convert QDQ ONNX model from scale=0 to 1

    convert QDQ ONNX model
    1. fold constants
    2. change scale to 1 if QDQ node has scale=0

    Parameters
    ----------
    onnx_model : onnx.ModelProto
        input onnx model to be converted
    logger : logging.Logger, default=None
        a logger

    Returns
    -------
    onverted_model : onnx.ModelProto
        converted onnx model
    """
    graph = gs.import_onnx(onnx_model)
    logger.debug('convert to graph done')

    graph.fold_constants().cleanup()
    logger.debug('constant folding done')

    _update_qdq_nodes(graph, logger)
    logger.debug('graph updating done')

    return gs.export_onnx(graph)


def _update_qdq_nodes(graph, logger):
    """update QDQ scales to 1 if node has scale=0

    Parameters
    ----------
    graph : onnx_graphsurgeon.Graph
        a graph imported from onnx
        this is inout parameter(it's updated in this function).
    logger : logging.Logger
        a logger
    """
    for node in graph.nodes:
        is_updated = _change_scale0to1(node)
        if is_updated:
            logger.info(f'{node.name}({node.op}): scale is changed to 1')


def _change_scale0to1(node):
    """if QuantizeLinear/DequantizeLinear node's scale is all 0, change scale to 1

    Parameters
    ---------
    node : onnx_graphsurgeon.Node
        a node
        this is inout parameter(it's updated in this function)

    Returns
    ------
    is_updated : bool
        True: node's inputs[1] is updated

    Examples
    --------
    only QuantizeLinear and DequantizeLinear is processed
    >>> node = gs.Node('Identity', inputs=[gs.Variable('input', np.float32, [1, 3, 122, 122])], outputs=[gs.Variable('output', np.float32, [1, 3, 122, 122])])
    >>> is_updated = _change_scale0to1(node)
    >>> is_updated
    False

    only all 0 scale is processed
    >>> scale_array = np.zeros(32, dtype=np.float32)
    >>> scale_array[2] = 0.5
    >>> node = gs.Node('QuantizeLinear', inputs=[gs.Variable('weight', np.float32, [32, 16, 1, 1]), gs.Constant('scale', scale_array)], outputs=[gs.Variable('quant_weight', np.int8, [32, 16, 1, 1])], attrs={'axis': 0})
    >>> is_updated = _change_scale0to1(node)
    >>> is_updated
    False
    >>> np.all(node.inputs[1].values == scale_array)  # keep original scale values
    True

    scalar scale=0 is converted to 1
    >>> node = gs.Node('QuantizeLinear', inputs=[gs.Variable('feature', np.float32, [1, 3, 122, 122]), gs.Constant('scale', np.zeros(1, dtype=np.float32))], outputs=[gs.Variable('quant_feature', np.int8, [1, 3, 122, 122])])
    >>> is_updated = _change_scale0to1(node)
    >>> is_updated
    True
    >>> (node.inputs[1].values == 1).item()
    True

    all 0 scale tensor is converted to all 1 tensor
    >>> node = gs.Node('DequantizeLinear', inputs=[gs.Variable('quant_weight', np.int8, [32, 16, 1, 1]), gs.Constant('scale', np.zeros(32, dtype=np.float32))], outputs=[gs.Variable('weight', np.float32, [32, 16, 1, 1])], attrs={'axis': 0})
    >>> is_updated = _change_scale0to1(node)
    >>>	is_updated
    True
    >>> np.all(node.inputs[1].values == 1)
    True
    >>> node.inputs[1].values.shape
    (32,)
    """
    if node.op not in ['QuantizeLinear', 'DequantizeLinear']:
        return False

    old_scale = node.inputs[1].values
    if not np.all(old_scale == 0):
        return False

    node.inputs[1].values = np.ones_like(old_scale)
    return True
