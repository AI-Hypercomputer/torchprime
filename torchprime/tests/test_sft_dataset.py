import json
from pathlib import Path
from unittest import mock

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from torchprime.data.sft_dataset import make_sft_dataset


def _write_json(tmpdir: Path, name: str, data):
  path = tmpdir / name
  with path.open("w") as f:
    for item in data:
      json.dump(item, f)
      f.write("\n")
  return path


def _tokenizer():
  vocab = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "Hello": 3,
    "World": 4,
    "Hi": 5,
    "Hey": 6,
    "Bye": 7,
    "See": 8,
  }
  model = WordLevel(vocab)
  tok = Tokenizer(model)
  tok.pre_tokenizer = Whitespace()
  return PreTrainedTokenizerFast(
    tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>"
  )


def test_local_json_prompt_completion(tmp_path: Path):
  data = [{"prompt": "Hello", "completion": "World"}]
  path = _write_json(tmp_path, "pc.json", data)
  tok = _tokenizer()
  ds = make_sft_dataset(tok, 16, json_path=str(path), format="prompt_completion")
  assert isinstance(ds, Dataset)
  assert ds[0]["labels"][0] == tok.encode("Hello", add_special_tokens=False)[0]


def test_gcp_json_chat_mask_last(tmp_path: Path):
  messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hey"},
  ]
  data = [{"messages": messages}]
  local = _write_json(tmp_path, "chat.json", data)
  tok = _tokenizer()
  with mock.patch("fsspec.open") as open_mock:
    open_mock.return_value.__enter__.return_value = local.open()
    ds = make_sft_dataset(
      tok,
      16,
      json_path="gs://bucket/chat.json",
      format="chat",
      mask_mode="last",
    )
  assert isinstance(ds, Dataset)
  user_ids = tok.encode("Hi", add_special_tokens=False)
  assistant_ids = tok.encode("Hey", add_special_tokens=False)
  labels = ds[0]["labels"]
  assert labels[len(user_ids) : len(user_ids) + len(assistant_ids)] == [-100] * len(
    assistant_ids
  )


def test_hf_dataset_pack_mask_all(tmp_path: Path):
  data = [
    {
      "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hey"},
      ]
    },
    {
      "messages": [
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "See"},
      ]
    },
  ]
  tok = _tokenizer()
  with mock.patch("torchprime.data.sft_dataset._load_hf_dataset") as loader:
    loader.return_value = Dataset.from_list(data)
    ds = make_sft_dataset(
      tok,
      4,
      hf_name="dummy",
      format="chat",
      mask_mode="all",
      pack_samples=True,
    )
  assert isinstance(ds, Dataset)
  assert len(ds) > 0
  # Assistant tokens masked
  labels = [x for x in ds[0]["labels"] if x != tok.pad_token_id]
  assert -100 in labels and any(x != -100 for x in labels)


def test_chat_multi_turn_mask_modes(tmp_path: Path):
  messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hey"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "World"},
  ]
  path = _write_json(tmp_path, "multi.json", [{"messages": messages}])
  tok = _tokenizer()

  modes = {
    "none": (False, False),
    "last": (False, True),
    "all": (True, True),
  }

  for mode, (mask_first, mask_second) in modes.items():
    ds = make_sft_dataset(
      tok,
      32,
      json_path=str(path),
      format="chat",
      mask_mode=mode,
    )
    labels = ds[0]["labels"]
    hi = tok.encode("Hi", add_special_tokens=False)
    hey = tok.encode("Hey", add_special_tokens=False)
    hello = tok.encode("Hello", add_special_tokens=False)
    world = tok.encode("World", add_special_tokens=False)
    idx0 = len(hi)
    idx1 = idx0 + len(hey)
    idx2 = idx1 + len(hello)
    idx3 = idx2 + len(world)
    assert labels[idx0:idx1] == ([-100] * len(hey) if mask_first else hey)
    assert labels[idx2:idx3] == ([-100] * len(world) if mask_second else world)
    eos_label = labels[idx3]
    if mode == "none":
      assert eos_label == tok.eos_token_id
    else:
      assert eos_label == -100
