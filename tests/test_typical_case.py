import onnx
import subprocess
import onnx.numpy_helper
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from qdqonnx_scale_0to1.qdq_scale_converter import convert_onnx_file


APP_NAME = 'qdq_scale0to1'


def _run_command(cmds):
    with subprocess.Popen(cmds, stdout=subprocess.PIPE) as proc:
        stdout_raw, stderr_raw = proc.communicate()
        ret = proc.returncode
    return ret, stdout_raw.decode('utf-8')


def test_cmmand():
    inputs = [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 3, 64, 64])]
    outputs = [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 16, 64, 64])]
    nodes = [
        onnx.helper.make_node('QuantizeLinear', ['input', 'scale0.scalar', 'zero_point'], ['input.q']),
        onnx.helper.make_node('DequantizeLinear', ['input.q', 'scale0.scalar', 'zero_point'], ['input.float']),
        onnx.helper.make_node('DequantizeLinear', ['weight.quant', 'scale0.vector', 'zero_point'], ['weight.float'], axis=0),
        onnx.helper.make_node('DequantizeLinear', ['bias.quant', 'scale0.scalar', 'bias.zero_point'], ['bias.float']),
        onnx.helper.make_node('Conv', ['input.float', 'weight.float', 'bias.float'], ['conv.output'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        onnx.helper.make_node('QuantizeLinear', ['conv.output', 'scale.output', 'zero_point'], ['conv.q']),
        onnx.helper.make_node('DequantizeLinear', ['conv.q', 'scale.output', 'zero_point'], ['output']),
    ]
    inits = [
        onnx.numpy_helper.from_array(np.zeros(1, dtype=np.float32), 'scale0.scalar'),
        onnx.numpy_helper.from_array(np.zeros(1, dtype=np.int8), 'zero_point'),
        onnx.numpy_helper.from_array(np.zeros(1, dtype=np.int32), 'bias.zero_point'),
        onnx.numpy_helper.from_array(np.ones([16, 3, 3, 3], dtype=np.int8), 'weight.quant'),
        onnx.numpy_helper.from_array(np.ones([16], dtype=np.int32), 'bias.quant'),
        onnx.numpy_helper.from_array(np.zeros([16], dtype=np.float32), 'scale0.vector'),
        onnx.numpy_helper.from_array(np.array(0.125, dtype=np.float32), 'scale.output'),
    ]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, 'qdq_with_all0scale', inputs, outputs, inits), opset_imports=[onnx.helper.make_opsetid('', 13)], ir_version=7)

    with TemporaryDirectory() as tmp_dir_path_name:
        input_filepath = Path(tmp_dir_path_name) / 'qdq_with_all0scale.onnx'
        output_filepath = Path(tmp_dir_path_name) / 'output.onnx'
        onnx.save(model, str(input_filepath))
        _run_command([APP_NAME, '-i', str(input_filepath), '-o', str(output_filepath)])
        result_onnx = onnx.load(str(output_filepath))

    initializer_dict = {ini.name: onnx.numpy_helper.to_array(ini) for ini in result_onnx.graph.initializer}
    assert np.all(initializer_dict[result_onnx.graph.node[0].input[1]] == 1)
    assert np.all(initializer_dict[result_onnx.graph.node[1].input[1]] == 1)
    assert np.all(initializer_dict[result_onnx.graph.node[2].input[1]] == 1)
    assert np.all(initializer_dict[result_onnx.graph.node[3].input[1]] == 1)


def test_convert_onnx_file():
    inputs = [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1])]
    outputs = [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1])]
    scale_value = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), 'const.tensor')
    zero_point_value = onnx.numpy_helper.from_array(np.array([0], dtype=np.int8), 'cos.tensor')
    nodes = [
        onnx.helper.make_node('Constant', [], ['const.scale'], value=scale_value),
        onnx.helper.make_node('ConstantOfShape', ['cos.scalar_shape'], ['cos.zero_point'], value=zero_point_value),
        onnx.helper.make_node('QuantizeLinear', ['input', 'const.scale', 'cos.zero_point'], ['model_input.q']),
        onnx.helper.make_node('DequantizeLinear', ['model_input.q', 'dq.scale'], ['output']),
    ]
    inits = [
        onnx.numpy_helper.from_array(np.empty(0, dtype=np.int64), 'cos.scalar_shape'),
        onnx.numpy_helper.from_array(np.array(0.25, dtype=np.float32), 'dq.scale'),
    ]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, 'qdq_with_constant', inputs, outputs, inits), opset_imports=[onnx.helper.make_opsetid('', 13)], ir_version=7)

    with TemporaryDirectory() as tmp_dir_path_name:
        input_filepath = Path(tmp_dir_path_name) / 'qdq_with_constant.onnx'
        output_filepath = Path(tmp_dir_path_name) / 'output.onnx'
        onnx.save(model, str(input_filepath))
        convert_onnx_file(input_filepath, output_filepath)
        result_onnx = onnx.load(str(output_filepath))

    # fold constants (Constant, ConstantOfShape nodes are converted to initializers)
    assert len(result_onnx.graph.node) == 2
    assert len(result_onnx.graph.initializer) == 3
    initializer_dict = {ini.name: onnx.numpy_helper.to_array(ini) for ini in result_onnx.graph.initializer}
    # QuantizeLinear's scale is not modify because it's not a all 0
    assert np.all(initializer_dict[result_onnx.graph.node[0].input[1]] == 0.5)
    # DequantizeLinear's scale is not modify because it's not a all 0
    assert np.all(initializer_dict[result_onnx.graph.node[1].input[1]] == 0.25)
