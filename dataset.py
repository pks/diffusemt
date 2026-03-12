import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_from_disk


class TranslationDataset(Dataset):
    def __init__(self, data_path="data/wmt14_en_de_tokenized"):
        self.ds = load_from_disk(data_path)
        self.ds.set_format("torch")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "source_ids": item["source_ids"],
            "source_mask": item["source_mask"].bool(),
            "target_ids": item["target_ids"],
            "target_mask": item["target_mask"].bool(),
        }


DATA_PATHS = {
    "train": "data/wmt14_en_de_bert_cased",
    "test": "data/wmt14_en_de_bert_cased_test",
}


def get_dataloader(config, split="train", distributed=False):
    dataset = TranslationDataset(data_path=DATA_PATHS[split])
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if distributed else None
    return DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler, num_workers=0, pin_memory=True,
    ), sampler
