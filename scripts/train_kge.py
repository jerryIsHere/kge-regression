from KGE_Regression.experiment import Pipeline
from pykeen.datasets import OpenBioLink


exp = Pipeline(
    OpenBioLink,
    "TransE_400_ep",
    cache_directory="/public/ckchan666/CS6536/cache",
    smile_path="/home/ms20/ckchan666/CS6536/217507403391658357.txt.gz",
    model_directory="/public/ckchan666/CS6536/model",
)
eppoch = 100
exp.train_full_graph(eppoch).save_full_graph(
    "TransE_400_ep" + str(eppoch)
).train_sub_graph(eppoch).save_sub_graph("TransE_400_ep" + str(eppoch)).build_pandas(
    embedding_smile_path="/public/ckchan666/CS6536/experinment/embedding_smile_df.pkl",
    masked_cid_smile_path="/public/ckchan666/CS6536/experinment/masked_cid_smile_df.pkl",
)
