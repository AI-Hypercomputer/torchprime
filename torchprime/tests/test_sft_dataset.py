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
    "<start>": 3,
    "<sep>": 4,
    "<end>": 5,
    "user": 6,
    "assistant": 7,
    "Hello": 8,
    "World": 9,
    "Hi": 10,
    "Hey": 11,
    "Bye": 12,
    "See": 13,
    "<unk>": 14,
  }
  model = WordLevel(vocab, unk_token="<unk>")
  tok = Tokenizer(model)
  tok.pre_tokenizer = Whitespace()
  tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
  )
  tokenizer.chat_template = "{% for message in messages %}<start> {{ message['role'] }} <sep> {{ message['content'] }} <end> {% endfor %}"
  return tokenizer


def test_chat_template():
  """Verify ``apply_chat_template`` tokenizes messages as expected."""

  tok = _tokenizer()
  messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hey"},
  ]
  text = tok.apply_chat_template(messages, tokenize=False)
  ids = [tok.convert_tokens_to_ids(t) for t in text.split()] + [tok.eos_token_id]
  expected = [
    tok.convert_tokens_to_ids("<start>"),
    tok.convert_tokens_to_ids("user"),
    tok.convert_tokens_to_ids("<sep>"),
    tok.convert_tokens_to_ids("Hi"),
    tok.convert_tokens_to_ids("<end>"),
    tok.convert_tokens_to_ids("<start>"),
    tok.convert_tokens_to_ids("assistant"),
    tok.convert_tokens_to_ids("<sep>"),
    tok.convert_tokens_to_ids("Hey"),
    tok.convert_tokens_to_ids("<end>"),
    tok.eos_token_id,
  ]
  assert ids == expected


def test_local_json_prompt_completion(tmp_path: Path):
  """Prompt/completion dataset tokenization.

  The JSON file contains one record with ``{"prompt": "Hello", "completion": "World"}``.
  Tokens ``Hello`` -> ``[8]`` and ``World`` -> ``[9]`` should appear in ``labels``
  with an EOS token appended and an ``attention_mask`` marking the real tokens.
  """

  data = [{"prompt": "Hello", "completion": "World"}]
  path = _write_json(tmp_path, "pc.json", data)
  tok = _tokenizer()
  ds = make_sft_dataset(
    file_dataset_path=str(path),
    format="prompt_completion",
    tokenizer=tok,
    block_size=16,
  )
  assert isinstance(ds, Dataset)
  # "Hello" -> [8], "World" -> [9], "eos" -> [2]
  assert ds[0]["input_ids"][:3] == [8, 9, 2]
  assert ds[0]["labels"][0] == -100  # masked prompt
  assert ds[0]["labels"][1] == tok.convert_tokens_to_ids("World")  # completion
  assert ds[0]["labels"][2] == 2  # eos
  mask = ds[0]["attention_mask"]
  assert len(mask) == 16
  assert mask[:3] == [1, 1, 1]
  assert all(m == 0 for m in mask[3:])


def test_gcp_json_chat_mask_last(tmp_path: Path):
  """Chat dataset from GCP with last assistant masking.

  The sample contains ``user: Hi`` followed by ``assistant: Hey``. All user
  tokens should be ``-100`` in ``labels`` while the assistant tokens remain
  unmasked.
  """

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
      file_dataset_path="gs://bucket/chat.json",
      format="chat",
      compute_loss_on="last_assistant",
      tokenizer=tok,
      block_size=16,
    )
  assert isinstance(ds, Dataset)
  user_ids = [
    tok.convert_tokens_to_ids("<start>"),
    tok.convert_tokens_to_ids("user"),
    tok.convert_tokens_to_ids("<sep>"),
    *tok.encode("Hi", add_special_tokens=False),
    tok.convert_tokens_to_ids("<end>"),
  ]
  assistant_ids = [
    tok.convert_tokens_to_ids("<start>"),
    tok.convert_tokens_to_ids("assistant"),
    tok.convert_tokens_to_ids("<sep>"),
    *tok.encode("Hey", add_special_tokens=False),
    tok.convert_tokens_to_ids("<end>"),
  ]
  input_ids = ds[0]["input_ids"]
  labels = ds[0]["labels"]
  assert input_ids[: len(user_ids) + len(assistant_ids)] == user_ids + assistant_ids
  assert input_ids[len(user_ids) + len(assistant_ids)] == 2  # eos token
  assert all(
    x == 0 for x in input_ids[(len(user_ids) + len(assistant_ids) + 1) :]
  )  # pad to block size
  assert labels[: len(user_ids)] == [-100] * len(user_ids)  # masked user tokens
  assert labels[len(user_ids) : len(user_ids) + len(assistant_ids)] == assistant_ids
  assert labels[len(user_ids) + len(assistant_ids)] == 2  # eos token
  assert all(
    x == -100 for x in labels[(len(user_ids) + len(assistant_ids) + 1) :]
  )  # mask padding
  assert ds[0]["attention_mask"][len(user_ids) + len(assistant_ids)] == 1


def test_hf_dataset_pack_mask_all(tmp_path: Path):
  """HF dataset loading and packing with assistant-only loss."""
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
  with mock.patch("torchprime.data.dataset._load_hf_dataset") as loader:
    loader.return_value = Dataset.from_list(data)
    ds = make_sft_dataset(
      hf_dataset_name="dummy",
      format="chat",
      compute_loss_on="assistant",
      pack_samples=True,
      tokenizer=tok,
      block_size=16,
    )
  assert isinstance(ds, Dataset)
  assert len(ds) > 0
  # User tokens masked
  labels = [x for x in ds[0]["labels"] if x != tok.pad_token_id]
  assert -100 in labels and any(x != -100 for x in labels)


def test_chat_multi_turn_mask_modes(tmp_path: Path):
  """Multi-turn chat masking options.

  Checks that masking ``none``, ``assistant`` and ``last_assistant`` behave as
  expected on two user/assistant pairs.
  """
  messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hey"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "World"},
  ]
  path = _write_json(tmp_path, "multi.json", [{"messages": messages}])
  tok = _tokenizer()

  modes = {
    "all": (False, False, False),
    "assistant": (False, False, True),
    "last_assistant": (True, False, True),
  }

  for mode, (mask_first, mask_second, mask_user) in modes.items():
    ds = make_sft_dataset(
      file_dataset_path=str(path),
      format="chat",
      compute_loss_on=mode,
      tokenizer=tok,
      block_size=32,
    )
    labels = ds[0]["labels"]
    hi = [
      tok.convert_tokens_to_ids("<start>"),
      tok.convert_tokens_to_ids("user"),
      tok.convert_tokens_to_ids("<sep>"),
      *tok.encode("Hi", add_special_tokens=False),
      tok.convert_tokens_to_ids("<end>"),
    ]
    hey = [
      tok.convert_tokens_to_ids("<start>"),
      tok.convert_tokens_to_ids("assistant"),
      tok.convert_tokens_to_ids("<sep>"),
      *tok.encode("Hey", add_special_tokens=False),
      tok.convert_tokens_to_ids("<end>"),
    ]
    hello = [
      tok.convert_tokens_to_ids("<start>"),
      tok.convert_tokens_to_ids("user"),
      tok.convert_tokens_to_ids("<sep>"),
      *tok.encode("Hello", add_special_tokens=False),
      tok.convert_tokens_to_ids("<end>"),
    ]
    world = [
      tok.convert_tokens_to_ids("<start>"),
      tok.convert_tokens_to_ids("assistant"),
      tok.convert_tokens_to_ids("<sep>"),
      *tok.encode("World", add_special_tokens=False),
      tok.convert_tokens_to_ids("<end>"),
    ]
    idx0 = len(hi)
    idx1 = idx0 + len(hey)
    idx2 = idx1 + len(hello)
    idx3 = idx2 + len(world)
    assert labels[:idx0] == ([-100] * len(hi) if mask_user else hi)
    assert labels[idx0:idx1] == ([-100] * len(hey) if mask_first else hey)
    assert labels[idx1:idx2] == ([-100] * len(hello) if mask_user else hello)
    assert labels[idx2:idx3] == ([-100] * len(world) if mask_second else world)
    eos_label = labels[idx3]
    assert eos_label == tok.eos_token_id


def test_truncation_modes(tmp_path: Path):
  """Truncation of overlong prompt/completion pairs."""
  data = [{"prompt": "Hello Hello Hello Hello", "completion": "World World"}]
  path = _write_json(tmp_path, "trunc.json", data)
  tok = _tokenizer()

  right = make_sft_dataset(
    file_dataset_path=str(path),
    format="prompt_completion",
    truncation="right",
    tokenizer=tok,
    block_size=5,
  )
  left = make_sft_dataset(
    file_dataset_path=str(path),
    format="prompt_completion",
    truncation="left",
    tokenizer=tok,
    block_size=5,
  )
  dropped = make_sft_dataset(
    file_dataset_path=str(path),
    format="prompt_completion",
    truncation="drop",
    tokenizer=tok,
    block_size=5,
  )

  assert len(dropped) == 0
  assert right[0]["input_ids"] == [8, 8, 8, 8, 9]  # drop the last "World" and eos
  assert left[0]["input_ids"][-1] == tok.eos_token_id


def test_samplewise_packing(tmp_path: Path):
  """Sample-wise packing without truncating individual examples."""
  data = [
    {"prompt": "Hello Hello Hello", "completion": "World World"},  # <-- sample 0
    {"prompt": "Hi", "completion": "Hey"},  # <-- sample 1
    {"prompt": "Bye", "completion": "See"},  # <-- sample 2
  ]
  path = _write_json(tmp_path, "pack.json", data)
  tok = _tokenizer()
  ds = make_sft_dataset(
    file_dataset_path=str(path),
    format="prompt_completion",
    pack_samples=True,
    tokenizer=tok,
    block_size=8,
  )
  assert len(ds) == 2  # sample 1 and 2 are packed into one
  ids_0 = (
    tok.encode("Hello", add_special_tokens=False)
    + tok.encode("Hello", add_special_tokens=False)
    + tok.encode("Hello", add_special_tokens=False)
    + tok.encode("World", add_special_tokens=False)
    + tok.encode("World", add_special_tokens=False)
    + [tok.eos_token_id]
  )
  assert ds[0]["input_ids"] == ids_0 + [tok.pad_token_id] * (8 - len(ids_0))

  ids_1 = (
    tok.encode("Hi", add_special_tokens=False)
    + tok.encode("Hey", add_special_tokens=False)
    + [tok.eos_token_id]
    + tok.encode("Bye", add_special_tokens=False)
    + tok.encode("See", add_special_tokens=False)
    + [tok.eos_token_id]
  )
  assert ds[1]["input_ids"] == ids_1 + [tok.pad_token_id] * (8 - len(ids_1))
