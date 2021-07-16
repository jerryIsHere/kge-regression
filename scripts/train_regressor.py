from KGE_Regression.datasets import build_train_test_dataset, inference_dataset
from KGE_Regression.model import RobertaForEmbeddingRegression
from KGE_Regression.kge_experiment import PathManager
from transformers import RobertaTokenizer

pm = PathManager(
    model_name="TransE_400",
    epoch=100,
    cache_directory="/public/ckchan666/CS6536/cache",
    smile_path="/home/ms20/ckchan666/CS6536/217507403391658357.txt.gz",
    model_directory="/public/ckchan666/CS6536/model",
    pandas_directory="/public/ckchan666/CS6536/pd",
)
embedding_smile_path, masked_cid_smile_path = pm.df_path()
tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
train_dataset, test_dataset = build_train_test_dataset(
    embedding_smile_path=embedding_smile_path, tokenizer=tokenizer
)

attention_loss_weight = 0.0025
model = RobertaForEmbeddingRegression.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1",
    hidden_dropout_prob=0.45,
    num_labels=400,
    attention_loss_weight=attention_loss_weight,
)
model.set_attention_loss_weight(attention_loss_weight)
from transformers import Trainer, TrainingArguments
import numpy as np


def evaluate_fn(eval):
    logits, labels = (eval.predictions, eval.label_ids)
    loss = np.mean(np.absolute(np.squeeze(logits) - labels))
    return {"eval_loss": loss}


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.0075,
    logging_dir="./logs/",
    evaluation_strategy="epoch",
    logging_steps=1,
    learning_rate=6e-05,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=evaluate_fn,
)
trainer.train()

from torch.utils.data import DataLoader
import numpy as np
import torch

loader = DataLoader(
    inference_dataset(masked_cid_smile_path, tokenizer),
    batch_size=64,
    shuffle=False,
    pin_memory=False,
)
embedding = np.empty([0, 400])
model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        output = model(batch["input_ids"].to(model.device))
        embedding = np.vstack(
            (embedding, np.squeeze(output.logits.to("cpu").detach().numpy()))
        )

import pandas as pd

masked_cid_smile = pd.read_pickle(masked_cid_smile_path)
masked_cid_smile_embedding = masked_cid_smile.copy()
masked_cid_smile_embedding["embedding"] = [row for row in embedding]
masked_cid_smile_embedding.to_pickle(pm.inference_df_path("convAtt"))
