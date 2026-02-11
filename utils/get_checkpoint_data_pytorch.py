import torch
from fire import Fire
from json import dumps, JSONEncoder

class Encoder(JSONEncoder):

    def default(self, o):
        if isinstance(o, torch.Tensor):
            pass
        return o.__dict__


def main(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = len(checkpoint["model_state_dict"])
    print(dumps(checkpoint, indent=4, cls=Encoder))

if __name__ == "__main__":
    Fire(main)