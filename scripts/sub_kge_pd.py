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
exp.load("sub_graph").build_pandas()
