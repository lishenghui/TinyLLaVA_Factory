import torch
import datasets
from datasets import load_dataset, concatenate_datasets


def extract_text(conversation: dict) -> list[dict]:
    messages = [
        {"from": f, "value": v}
        for f, v in zip(conversation["from"], conversation["value"])
    ]
    return messages


def preprocess_batch_hf(examples, text_preprocess, image_processor, is_multimodal=True):

    input_ids = []
    labels = []
    images = []

    for i in range(len(examples["conversations"])):
        conv = examples["conversations"][i]  # dict
        # text = extract_text(conv)
        text = conv
        data_dict = text_preprocess(text)  # 文本处理
        if "image_raw" in examples and examples["image_raw"][i] is not None:
            image = examples["image_raw"][i]  # PIL.Image 或 HF Image
            image = image.convert("RGB")
            image_tensor = image_processor(image, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            # data_dict["image"] = image_tensor
        elif is_multimodal:
            raise ValueError("Image is required for multimodal input.")

        input_ids.append(torch.tensor(data_dict["input_ids"], dtype=torch.long))
        labels.append(torch.tensor(data_dict["labels"], dtype=torch.long))
        images.append(image_tensor)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "image": images,
    }


def get_train_dataset(text_preprocess, image_processor, is_multimodal=True):
    dataset = "fedlib/TinyCOCO-Data"
    dataset = load_dataset(dataset, split="train")
    dataset = dataset.rename_column("image", "image_raw")

    # dataset_path = "fedlib/TinyLLAVA-Training-Data"
    # coco_stream = load_dataset(dataset_path, split="coco", streaming=True)
    # coco_samples = list(coco_stream.take(1000))
    # dataset = datasets.Dataset.from_list(coco_samples)
    # coco_stream = load_dataset(dataset, split="coco", streaming=True)
    # textvqa_stream = load_dataset(dataset, split="textvqa", streaming=True)

    # coco_subset = Dataset.from_generator(lambda: coco_stream.take(5000))
    # textvqa_subset = Dataset.from_generator(lambda: textvqa_stream.take(5000))

    # coco_samples = list(coco_stream.take(1000))
    # textvqa_samples = list(textvqa_stream.take(1000))

    # coco_subset = datasets.Dataset.from_list(coco_samples)
    # textvqa_subset = datasets.Dataset.from_list(textvqa_samples)

    # dataset = concatenate_datasets([coco_subset, textvqa_subset])
    # dataset = concatenate_datasets([coco_subset])
    # dataset = dataset.shuffle(seed=43)
    # dataset = dataset.rename_column("image", "image_raw")

    train_set = dataset.map(
        lambda ex: preprocess_batch_hf(
            ex, text_preprocess, image_processor, is_multimodal=True
        ),
        batched=True,
        batch_size=16,
        num_proc=16,
    )
    train_set.set_format(type="torch", columns=["input_ids", "labels", "image"])
    return train_set
