import logging
import sys
import unittest

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr
from torch_xla._internal import tpu
from torch_xla.distributed.spmd import Mesh

from torchprime.torch_xla_models.experimental.custom_kernel import (
  SplashAttentionConfig,
  splash_attention,
)

if xr.device_type() == 'TPU':
  from torch_xla.experimental.custom_kernel import jax_import_guard
  jax_import_guard()
  import jax


def with_jax_high_precision(func):

  def wrapper(*args, **kwargs):
    jax.config.update('jax_default_matmul_precision', "highest")
    try:
      result = func(*args, **kwargs)
    finally:
      jax.config.update('jax_default_matmul_precision', "default")
    return result

  return wrapper

class SplashAttentionTest(unittest.TestCase):

  def setUp(self):
    ### Splash attention block sizes
    # These can be tuned for specific hardware generations, and can be set up to
    # the model's sequence length.
    self.config = SplashAttentionConfig(
      sa_block_q=512,
      sa_block_kv=512,
      sa_block_kv_compute=512,
      sa_block_q_dkv=512,
      sa_block_kv_dkv=512,
      sa_block_kv_dkv_compute=512,
      sa_block_q_dq=512,
      sa_block_kv_dq=512,
      sa_use_fused_bwd_kernel=False,
      sa_q_layout="HEAD_DIM_MINOR",
      sa_k_layout="HEAD_DIM_MINOR",
      sa_v_layout="HEAD_DIM_MINOR",
      mesh = str(xs.get_global_mesh())
    )
    # Common dimensions for all tests. NUM_HEADS, SEQ_LEN, HEAD_DIM must >= 128
    # for splash attention kernel.
    self.BATCH_SIZE = 4
    self.NUM_HEADS = 128
    self.SEQ_LEN = 128
    self.HEAD_DIM = 128

  def _attention(self, q, k, v, *, attn_mask=None, ab=None):
    # q shape: [batch, #head, seq_len, head_dim]
    attn_weight = q @ k.transpose(-2, -1)
    if attn_mask is not None:
      # Masked out the unrelevant parts.
      attn_weight = attn_weight.masked_fill(attn_mask,
                                            torch.finfo(attn_weight.dtype).min)
    if ab is not None:
      attn_weight = attn_weight + ab
    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
    attn_output = attn_weight @ v
    return attn_output


  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_base(self):

    q = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    k = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    v = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    q_sa = q.clone().detach().requires_grad_(True)
    k_sa = k.clone().detach().requires_grad_(True)
    v_sa = v.clone().detach().requires_grad_(True)
    attention_mask = torch.triu(torch.ones(self.SEQ_LEN, self.SEQ_LEN), diagonal=1).to("xla")

    o =  self._attention(q, k, v, attn_mask=attention_mask)
    torch_xla.sync()
    q.retain_grad()
    loss = torch.sum(o)
    loss.backward()
    torch_xla.sync()
    q_grad = q.grad


    o_sa = splash_attention(q_sa,k_sa,v_sa,self.config.to_json())
    torch_xla.sync()
    q_sa.retain_grad()
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    torch_xla.sync()
    q_grad_sa = q_sa.grad

    torch.testing.assert_close(o, o_sa, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(q_grad.cpu(), q_grad_sa.cpu(), rtol=1e-2, atol=1e-3)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_sharding(self):

    q = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    k = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    v = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    o = splash_attention(q, k, v, self.config.to_json())
    torch_xla.sync()
    #TODO: Currently the output is `{replicated}`. Check why Maxtext use
    #replicated in Splash attention kernel by default? Or maybe we are wrong
    #about what Maxtext is doing.
    print(torch_xla._XLAC._get_xla_sharding_spec(o))


  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_segment_id(self):
    # test the segment id in splash attention against the flash attention kernel
    from torch_xla.experimental.custom_kernel import flash_attention

    q = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    k = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    v = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    q_sa = q.clone().detach().requires_grad_(True)
    k_sa = k.clone().detach().requires_grad_(True)
    v_sa = v.clone().detach().requires_grad_(True) 

    # make sure query.shape[2] == decoder_segment_ids.q.shape[1]
    segment_ids = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN).to("xla")
    for i in range(self.BATCH_SIZE):
        segment_ids[i, :] = i  # each batch item is in its own segment

    partition_spec=("data", None, None, None)
    o = flash_attention(
        q, k, v, True, segment_ids.to("xla"), segment_ids.to("xla"), partition_spec=partition_spec, mesh=xs.get_global_mesh())
    torch_xla.sync()
    q.retain_grad()
    loss = torch.sum(o)
    loss.backward()
    torch_xla.sync()
    q_grad = q.grad

    o_sa = splash_attention(q_sa,k_sa,v_sa,self.config.to_json(), decoder_segment_ids=segment_ids)
    torch_xla.sync()
    q_sa.retain_grad()
    loss_sa = torch.sum(o_sa)
    loss_sa.backward()
    torch_xla.sync()
    q_grad_sa = q_sa.grad

    torch.testing.assert_close(o, o_sa, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(q_grad.cpu(), q_grad_sa.cpu(), rtol=1e-4, atol=1e-5)
   


  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @with_jax_high_precision
  def test_splash_attention_aot_traceable(self):
    from functorch.compile import aot_function, make_boxed_func

    def compiler(gm, _):
      return make_boxed_func(gm)

    compiled_splash_attention = aot_function(
        splash_attention, fw_compiler=compiler)
    q = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    k = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")
    v = torch.randn(self.BATCH_SIZE, self.NUM_HEADS, self.SEQ_LEN, self.HEAD_DIM).requires_grad_(True).to("xla")

    segment_ids = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN).to("xla")
    for i in range(self.BATCH_SIZE):
        segment_ids[i, :] = i  
    o = compiled_splash_attention(
        q, k, v, config=self.config.to_json(), decoder_segment_ids=segment_ids)
    print(o)
    q.retain_grad()
    loss = o.sum()
    loss.backward()
    torch_xla.sync()
    print(q.grad)


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  torch.manual_seed(42)
  xr.use_spmd()
  partition_spec = ('data', 'fsdp', None, None)
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (num_devices // 2, 2)
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ('data', 'fsdp'))
  xs.set_global_mesh(mesh)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
