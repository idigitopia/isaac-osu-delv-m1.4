import os

import numpy as np
import onnx
import onnxruntime as ort


class OnnxPolicy:
    def __init__(self, policy_path):
        policy = onnx.load(os.path.join(policy_path, "policy.onnx"))
        onnx.checker.check_model(policy)
        print("Loaded and verified ONXX policy.")

        providers = ["CUDAExecutionProvider"]
        self._session = ort.InferenceSession(os.path.join(policy_path, "policy.onnx"), providers=providers)
        print("ONNX Runtime inference session created successfully.")

        input_args_length = len(self._session.get_inputs())
        self._input_names = [self._session.get_inputs()[i].name for i in range(input_args_length)]
        self._output_name = self._session.get_outputs()[0].name

    def __call__(self, **kwargs):
        input_obs = {k: np.expand_dims(v, axis=0) for k, v in kwargs.items()}
        results = self._session.run([self._output_name], input_obs)
        return results[0].squeeze()
