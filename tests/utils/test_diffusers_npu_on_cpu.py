# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

pytest.importorskip("diffusers")


def test_diffusers_affine_free_rms_norm_uses_native_math_on_npu(monkeypatch):
    from diffusers.models.normalization import RMSNorm
    from verl.utils import device
    from verl_omni.utils.diffusers_npu import apply_diffusers_npu_rms_norm_patch

    original_forward = RMSNorm.forward
    monkeypatch.setattr(RMSNorm, "forward", original_forward)
    monkeypatch.setattr(device, "get_device_name", lambda: "npu")
    apply_diffusers_npu_rms_norm_patch()

    norm = RMSNorm(8, eps=1e-6, elementwise_affine=False)
    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    expected = hidden_states * torch.rsqrt(
        hidden_states.float().pow(2).mean(-1, keepdim=True) + norm.eps
    ).to(hidden_states.dtype)

    output = norm(hidden_states)

    assert output.dtype == hidden_states.dtype
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
