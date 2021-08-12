from KGE_Regression.kge_experiment import Pipeline, PathManager
from pykeen.datasets import OpenBioLink

pm = PathManager(
    model_name="RESCAL",
    cache_directory="/public/ckchan666/CS6536/cache",
    smile_path="/home/ms20/ckchan666/CS6536/217507403391658357.txt.gz",
    model_directory="/public/ckchan666/CS6536/model",
    pandas_directory="/public/ckchan666/CS6536/pd",
)
exp = Pipeline(OpenBioLink, pm)
embedding_smile_path, masked_cid_smile_path = pm.df_path()
exp.train_sub_graph()
exp.evaluate_sub_graph()
for group in exp.results:
    print(group)
    print(exp.results[group].to_df().to_string())
    print()

exp.build_pandas()
