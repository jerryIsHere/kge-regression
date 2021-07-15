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
embedding_smile_path, masked_cid_smile_path = pm.df_path()
exp.train_full_graph().save_full_graph().train_sub_graph().save_sub_graph().build_pandas(
    embedding_smile_path=embedding_smile_path,
    masked_cid_smile_path=masked_cid_smile_path,
)
