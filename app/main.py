#!python3.10
import io
from fastapi import FastAPI, File, UploadFile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi.responses import JSONResponse
import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args

import Models

MODEL_REPO_MAP = {
    "eva02_large": "SmilingWolf/wd-eva02-large-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "vit_large": "SmilingWolf/wd-vit-large-tagger-v3",
    "swinv2_v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "swinv2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


@flax.struct.dataclass
class PredModel:
    apply_fun: Callable = flax.struct.field(pytree_node=False)
    params: Any = flax.struct.field(pytree_node=True)

    def jit_predict(self, x):
        # Not actually JITed since this is a single shot script,
        # but this is the function you would decorate with @jax.jit
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = flax.linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        preds = preds[0]
        return preds


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    # Resize
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize(
            (target_size, target_size),
            Image.BICUBIC,
        )
    return image


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"selected_tags.csv failed to download from {repo_id}"
        ) from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def load_model_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> PredModel:
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.msgpack",
        revision=revision,
        token=token,
    )

    model_config = hf_hub_download(
        repo_id=repo_id,
        filename="sw_jax_cv_config.json",
        revision=revision,
        token=token,
    )

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = flax.serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(
        config=model_builder,
        **model_config["model_args"],
    )
    model = PredModel(model.apply, params=variables)
    return model, model_config["image_size"]


def get_tags(
    probs: Any,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(
            gen_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(
            char_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@dataclass
class ScriptOptions:
    image_file: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.35)


app = FastAPI()

repo_id = MODEL_REPO_MAP.get("swinv2_v3")
print(f"Loading model 'swinv2_v3' from '{repo_id}'...")
model, target_size = load_model_hf(repo_id=repo_id)

print("Loading tag list...")
labels: LabelData = load_labels_hf(repo_id=repo_id)


@app.get("/")
def read_root():
    return "Deepdanbooru"


@app.post("/predict")
async def read_item(image: UploadFile = File(...)):
    contents = await image.read()

    print("Loading image and preprocessing...")
    try:
        image = Image.open(io.BytesIO(contents))
        img_input: Image.Image = Image.open(io.BytesIO(contents))
        img_input = pil_ensure_rgb(img_input)
        img_input = pil_pad_square(img_input)
        img_input = pil_resize(img_input, target_size)
        inputs = np.array(img_input)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = inputs[..., ::-1]
        print("Running inference...")
        outputs = model.predict(inputs)

        print("Processing results...")
        caption, taglist, ratings, character, general = get_tags(
            probs=outputs,
            labels=labels,
            gen_threshold=ScriptOptions.gen_threshold,
            char_threshold=ScriptOptions.char_threshold,
        )
        tags = [x for x in general]
        tags.extend([x for x in character])
        return JSONResponse(content={
            "ratings": [{"name": k, "score": float(v)} for k, v in ratings.items()],
            "character": [{"name": k, "score": float(v)} for k, v in character.items()],
            "general": [{"name": k, "score": float(v)} for k, v in general.items()],
            "tags": tags
        }, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)
