from pathlib import Path
import json
import re
import torch
from transformers import AutoTokenizer

data_dir = Path("results/alfworld/hmm_data_9b")  # change this
meta_path = next(data_dir.glob("*.metadata.json"))
meta = json.loads(meta_path.read_text())

dataset = meta["dataset"]
eos_id = meta["eos_token_id"]
tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer"])

files = [
    data_dir / f"{dataset}.dev",
    data_dir / f"{dataset}.lvd",
    *sorted(data_dir.glob(f"{dataset}.train.*")),
]

expected = re.compile(
    r"^\s*<state>\{[^\r\n]*\}</state>\s*"
    r"<action>.+</action>\s*$"
)

print(json.dumps(meta, indent=2))

for path in files:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    print(f"\n{path.name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    for index, row in enumerate(tensor[:5]):
        ids = row.tolist()

        # The first EOS is the real terminator; subsequent EOS tokens are padding.
        stop = ids.index(eos_id)
        text = tokenizer.decode(
            ids[:stop],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        print(f"\n--- row {index}; tokens={stop}; format_ok={bool(expected.fullmatch(text))}")
        print(text)