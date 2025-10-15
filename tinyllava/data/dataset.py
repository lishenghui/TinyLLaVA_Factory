from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence

import torch
import transformers
from PIL import Image, ImageFile

from tinyllava.data.get_training_data import get_train_dataset

from ..utils.constants import *
from .text_preprocess import TextPreprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        input_ids = [
            torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x
            for x in input_ids
        ]
        labels = [
            torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x
            for x in labels
        ]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, : self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if "image" in instances[0]:
            images = [
                torch.tensor(instance["image"], dtype=torch.float32)
                for instance in instances
            ]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args  # , text_preprocess
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
    train_dataset = get_train_dataset(
        text_preprocess,
        data_args.image_processor,
        is_multimodal=True,
    )

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
