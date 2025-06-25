import json
from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from torchprime.data.dpo_dataset import make_dpo_dataset


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
    "Hey": 5,
    "<unk>": 6,
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
  return tokenizer


def test_local_json_pair(tmp_path: Path):
  data = [
    {"prompt": "Hello", "chosen": "World", "rejected": "Hey"},
  ]
  path = _write_json(tmp_path, "pairs.json", data)
  tok = _tokenizer()
  ds = make_dpo_dataset(file_dataset_path=str(path), tokenizer=tok, block_size=8)
  assert isinstance(ds, Dataset)
  rec = ds[0]
  hello_id = tok.convert_tokens_to_ids("Hello")
  world_id = tok.convert_tokens_to_ids("World")
  hey_id = tok.convert_tokens_to_ids("Hey")
  eos = tok.eos_token_id
  assert rec["chosen_input_ids"][:3] == [hello_id, world_id, eos]
  assert rec["chosen_labels"][0] == -100
  assert rec["chosen_labels"][1] == world_id
  assert rec["rejected_input_ids"][:3] == [hello_id, hey_id, eos]
  assert rec["rejected_labels"][1] == hey_id
