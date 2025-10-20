import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from transformers import AutoTokenizer

from tinyllava.data.dataset import (
    LazySupervisedDataset,
    make_supervised_data_module,
    make_supervised_data_module_hf,
    DataCollatorForSupervisedDataset,
)
from tinyllava.utils import DataArguments


class TestDatasetParity(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 2048

        # Create a temporary directory for data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_folder = self.temp_dir.name

        # Create a dummy image
        self.image_file = "test_image.png"
        image_path = os.path.join(self.image_folder, self.image_file)
        Image.new("RGB", (10, 10), color="red").save(image_path)

        # Create dummy data
        self.sample_data = [
            {
                "image": self.image_file,
                "conversations": [
                    {"from": "human", "value": "<image>\nWhat is this?"},
                    {"from": "gpt", "value": "This is a test image."},
                ],
            },
            {
                "conversations": [
                    {"from": "human", "value": "Hello there."},
                    {"from": "gpt", "value": "General Kenobi!"},
                ]
            },
        ]

        # Create a temporary JSON file for the dataset
        self.data_path = os.path.join(self.temp_dir.name, "test_data.json")
        with open(self.data_path, "w") as f:
            json.dump(self.sample_data, f)

        # Mock image processor
        self.mock_image_processor = MagicMock()
        self.mock_image_processor.crop_size = {"height": 384, "width": 384}
        # A real image processor returns a dict with a 4D 'pixel_values' tensor (batch, C, H, W)
        # The code under test then selects the first image from the batch.
        self.mock_image_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 384, 384)
        }

        # Set up DataArguments
        data_args = DataArguments(
            data_path=self.data_path,
            lazy_preprocess=True,
            is_multimodal=True,
            image_folder=self.image_folder,
            image_aspect_ratio="square",
            conv_version="llama",
        )
        data_args.image_processor = self.mock_image_processor
        self.data_args = data_args

    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()

    @patch("tinyllava.data.dataset.get_train_dataset")
    def test_dataset_parity(self, mock_get_train_dataset):
        """Test that both data module functions produce identical data."""
        # To ensure parity, we first create the LazySupervisedDataset and use it
        # to generate the processed data. Then, we mock get_train_dataset to return
        # a Hugging Face Dataset created from this already-processed data.
        lazy_dataset = LazySupervisedDataset(
            tokenizer=self.tokenizer, data_path=self.data_path, data_args=self.data_args
        )
        processed_data = [lazy_dataset[i] for i in range(len(lazy_dataset))]

        # Mock `get_train_dataset` to return a dataset with pre-processed items.
        hf_dataset = HFDataset.from_list(processed_data)
        mock_get_train_dataset.return_value = hf_dataset

        # Create datasets using both functions
        data_module1 = make_supervised_data_module(self.tokenizer, self.data_args)
        # Pass the created hf_dataset to satisfy the new function signature
        data_module2 = make_supervised_data_module_hf(
            self.tokenizer, self.data_args, dataset=hf_dataset
        )

        dataset1 = data_module1["train_dataset"]
        dataset2 = data_module2["train_dataset"]

        # Ensure the HF dataset returns tensors, to match LazySupervisedDataset's behavior
        dataset2.set_format("torch")

        self.assertEqual(len(dataset1), len(dataset2))

        # Compare each item from both datasets
        for i in range(len(dataset1)):
            item1 = dataset1[i]
            item2 = dataset2[i]
            # Check that input_ids and labels are the same
            self.assertTrue(torch.equal(item1["input_ids"], item2["input_ids"]))
            self.assertTrue(torch.equal(item1["labels"], item2["labels"]))
            # Check image presence and tensor equality
            self.assertEqual("image" in item1, "image" in item2)
            if "image" in item1:
                self.assertTrue(torch.equal(item1["image"], item2["image"]))

    @patch("tinyllava.data.dataset.get_train_dataset")
    def test_to_hf_dataset_conversion(self, mock_get_train_dataset):
        """
        Tests the to_hf_dataset method and its integration with
        make_supervised_data_module_hf.
        """
        # 1. Create a standard LazySupervisedDataset as a baseline
        data_module_lazy = make_supervised_data_module(self.tokenizer, self.data_args)
        dataset_lazy = data_module_lazy["train_dataset"]

        # 2. Use to_hf_dataset to convert it to a Hugging Face Dataset
        hf_dataset = dataset_lazy.to_hf_dataset()

        # 3. Create a dataset using the HF-based data module function
        # We need to mock get_train_dataset to use the hf_dataset we just created
        processed_data = [dataset_lazy[i] for i in range(len(dataset_lazy))]
        mock_get_train_dataset.return_value = HFDataset.from_list(processed_data)

        data_module_hf = make_supervised_data_module_hf(
            self.tokenizer, self.data_args, dataset=hf_dataset
        )

        dataset_hf = data_module_hf["train_dataset"]
        dataset_hf.set_format("torch")

        # 4. Assert that both datasets are identical
        self.assertEqual(len(dataset_lazy), len(dataset_hf))
        for i in range(len(dataset_lazy)):
            self.assertTrue(
                torch.equal(dataset_lazy[i]["input_ids"], dataset_hf[i]["input_ids"])
            )
            self.assertTrue(
                torch.equal(dataset_lazy[i]["labels"], dataset_hf[i]["labels"])
            )
