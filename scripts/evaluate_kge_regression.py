from KGE_Regression.kge_experiment import Pipeline, PathManager
from pykeen.datasets import OpenBioLink


pm = PathManager(
    model_name="TransE_400",
    epoch=100,
    cache_directory="/public/ckchan666/CS6536/cache",
    smile_path="/home/ms20/ckchan666/CS6536/217507403391658357.txt.gz",
    model_directory="/public/ckchan666/CS6536/model",
    pandas_directory="/public/ckchan666/CS6536/pd",
)
exp = Pipeline(OpenBioLink, pm)
import pandas as pd

convAtt_masked_cid_smile_embedding = pd.read_pickle(pm.inference_df_path("convAtt"))
MLP_masked_cid_smile_embedding = pd.read_pickle(pm.inference_df_path("mlp"))
convAtt_masked_cid_smile_embedding.cid = convAtt_masked_cid_smile_embedding.cid.astype(
    int
)
MLP_masked_cid_smile_embedding.cid = MLP_masked_cid_smile_embedding.cid.astype(int)

import torch
import numpy as np


def convAtt_mapping(entity):
    try:
        return torch.Tensor(
            convAtt_masked_cid_smile_embedding.embedding[
                convAtt_masked_cid_smile_embedding.cid == int(entity.split(":")[-1])
            ].iloc[0]
        )
    except:
        print(entity)
        return torch.Tensor(np.zeros(400))


def MLP_mapping(entity):
    try:
        return torch.Tensor(
            MLP_masked_cid_smile_embedding.embedding[
                MLP_masked_cid_smile_embedding.cid == int(entity.split(":")[-1])
            ].iloc[0]
        )
    except:
        print(entity)
        return torch.Tensor(np.zeros(400))


exp.load("sub_graph").load("full_graph")
name = "com_convAtt"
exp.build_complemented_model(convAtt_mapping, name)
exp.evaluate_complemented_model(name)
del exp.models[name]

name = "com_MLP"
exp.build_complemented_model(MLP_mapping, name)
exp.evaluate_complemented_model(name)
del exp.models[name]

name = "com_full"
exp.build_complemented_model(
    lambda e: exp.models["full_graph"].entity_embeddings()[
        exp.dataset.training.entity_to_id[e]
    ],
    name,
)
exp.evaluate_complemented_model(name)
del exp.models[name]

name = "com_random"
exp.build_complemented_model(name=name)
exp.evaluate_complemented_model(name)
del exp.models[name]

exp.load("full_graph")
exp.evaluate_full_graph()
del exp.models["full_graph"]

exp.load("sub_graph")
exp.evaluate_sub_graph()
del exp.models["sub_graph"]

for group in exp.results:
    print(group)
    print(exp.results[group].to_df().to_string())
    print()

exp.plot_results("/home/ms20/ckchan666/CS6536/plt").plot_results(
    "/home/ms20/ckchan666/CS6536/plt", ["sub_graph"]
)
