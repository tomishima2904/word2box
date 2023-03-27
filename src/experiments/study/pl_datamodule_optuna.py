import lightning.pytorch as pl

from language_modeling_with_boxes.datasets.utils import \
    load_train_data_as_tensor, get_train_iter


class MyDataModule(pl.LightningDataModule):
    def __init__(self, n_gram, trial, config, vocab) -> None:
        super().__init__()
        self.data_dir = config["dataset"]
        self.batch_size = config["batch_size"]

        self.n_gram = n_gram
        self.config = config
        self.vocab = vocab

        self.subsample_thresh = trial.suggest_float("subsample_thresh", 1e-4, 1, log=True)


    def setup(self):
        self.train_tokenized = load_train_data_as_tensor(self.datadir)


    def train_dataloder(self):
        return get_train_iter(
                    self.config["batch_size"],
                    self.config["dataset"],
                    self.config["model_type"],
                    self.n_gram,
                    self.subsample_thresh,
                    self.config["data_device"],
                    self.config["add_pad"],
                    self.config["eos_mask"],
                    self.config["ignore_unk"],
                    self.vocab,
                )
