import wandb
import os
import time
import torch
from pneumoniaclassifier.model import _build_model
logdir="tmp_model"

def load_model(model_name):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_name)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    model = _build_model(model_name="efficientnet_b0", num_classes=2, pretrained=True)
    state_dict = torch.load(os.path.join(logdir, file_name), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    model.eval()
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1
