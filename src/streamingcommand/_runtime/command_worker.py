"""Owned command-model JSON-lines process; no application or experiment imports."""
from contextlib import redirect_stdout
import json
import os
import sys
import traceback
from pathlib import Path
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "streamingcommand._runtime"
from . import device_policy


def main():
    expected = device_policy.pin_device()
    runtime = None
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.pop('op')
            with redirect_stdout(sys.stderr):
                if operation == 'load':
                    import torch
                    from .single_command_runtime import SingleCommandRuntime
                    torch.set_num_threads(4)
                    if expected and device_policy.DEVICE.startswith('cuda:'):
                        actual = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
                        if actual != expected.lower().removeprefix('gpu-'):
                            raise RuntimeError('GPU UUID mismatch')
                    runtime = SingleCommandRuntime(request['checkpoint'], model_id=request['model'],
                        base_model_dir=request.get('base_model_dir'), use_cuda_graphs=request.get('use_cuda_graphs', True))
                    runtime.warmup(request['candidates'])
                    result = {**runtime.identity, 'model_id': request['model'], 'device': device_policy.DEVICE,
                              'device_uuid': expected, 'backbones': 1, 'adapters': 1, 'adapter_merged': False}
                elif operation == 'close':
                    result = {}
                elif runtime is None:
                    raise RuntimeError('Load the command model first')
                elif operation == 'parse':
                    result = runtime.parse_turn(**request)
                elif operation == 'parse_pair':
                    result = runtime.parse_pair(**request)
                elif operation == 'feedback':
                    result = runtime.feedback(**request)
                elif operation in ('clear', 'end'):
                    if runtime.session_id == request.get('session_id'):
                        runtime.clear()
                        if operation == 'end':
                            runtime.session_id = None
                    result = {}
                else:
                    raise ValueError('Unknown operation: ' + operation)
            print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False), file=protocol, flush=True)
            if operation == 'close':
                return
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False), file=protocol, flush=True)


if __name__ == '__main__':
    main()
