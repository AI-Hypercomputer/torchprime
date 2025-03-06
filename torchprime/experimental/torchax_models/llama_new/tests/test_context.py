# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import unittest

import torch

from models.llama4.reference_impl.context import (
    ContextManager,
    get_inference_context,
    InferenceContext,
)


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        context = get_inference_context()
        if context.hidden_states is not None:
            return x + context.hidden_states
        return x


class ContextTests(unittest.TestCase):

    def test_context(self):
        hidden_states = torch.randn(32, 32)
        llm_context = InferenceContext(
            tokens_position=TokensPosition(0),
            mask=LLMMask(),
            hidden_states=hidden_states,
        )
        model = ExampleModel()
        x = torch.randn(32, 32)
        with ContextManager(llm_context):
            output = model(x)
        torch.testing.assert_close(output, x + hidden_states)
