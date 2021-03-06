import os
import numpy as np
import pandas as pd
from functools import reduce
from pykeen.hpo import hpo_pipeline_from_config
from optuna.samplers import TPESampler


# Model_dict = {
#     "TransE_400": lambda triples_factory: TransE(
#         triples_factory=triples_factory, embedding_dim=400, random_seed=1234
#     ),
#     "RESCAL_400": lambda triples_factory: RESCAL(
#         triples_factory=triples_factory, embedding_dim=400, random_seed=1234
#     ),
#     "RESCAL_100": lambda triples_factory: RESCAL(
#         triples_factory=triples_factory, embedding_dim=100, random_seed=1234
#     ),
#     "RESCAL_50": lambda triples_factory: RESCAL(
#         triples_factory=triples_factory, embedding_dim=50, random_seed=1234
#     ),
# }


class PathManager:
    def __init__(
        self,
        model_name,
        smile_path,
        cache_directory,
        model_directory,
        pandas_directory,
    ):
        self.model_name = model_name
        self.smile_path = smile_path
        self.cache_directory = cache_directory
        self.model_directory = model_directory
        self.pandas_directory = pandas_directory

    def model_describer(
        self,
    ):
        return self.model_name

    def model_path(self, key, trial):
        return os.path.join(self.model_dir_path(key), str(trial), "trained_model.pkl")

    def model_dir_path(self, key):
        return os.path.join(self.model_directory, key + "_" + self.model_describer())

    def df_path(self):
        return (
            os.path.join(
                self.pandas_directory,
                self.model_describer() + "_" + "embedding_smile_df.pkl",
            ),
            os.path.join(
                self.pandas_directory,
                self.model_describer() + "_" + "masked_cid_smile_df.pkl",
            ),
        )

    def inference_df_path(self, regressor_name):
        return os.path.join(
            self.pandas_directory,
            self.model_describer() + "_" + regressor_name + "_embedding_smile_df.pkl",
        )


default_config = {
    "optuna": dict(
        n_trials=50,
    ),
    "pipeline": dict(
        model_kwargs=dict(random_seed=1234),
        model_kwargs_ranges=dict(
            embedding_dim=dict(type="categorical", choices=[300]),
        ),
        optimizer="SGD",
        optimizer_kwargs_ranges=dict(
            lr=dict(type="categorical", choices=[0.05, 0.045, 0.0475, 0.055, 0.0525])
        ),
        regularizer="LpRegularizer",
        regularizer_kwargs_ranges=dict(
            weight=dict(type="categorical", choices=[3e-7]),
            p=dict(type="categorical", choices=[1, 2]),
        ),
        loss="BCEAfterSigmoidLoss",
        training_loop="slcwa",
        training_kwargs=dict(
            batch_size=2048,
            num_epochs=3000,
        ),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=1),
        evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(batch_size=32),
        stopper="early",
        stopper_kwargs=dict(evaluation_batch_size=32, patience=5, frequency=100),
    ),
}


class Pipeline:
    def __init__(
        self,
        dataset,
        path_manager: PathManager,
        config=default_config,
        optiizer="SGD",
        ks=[1, 3, 5, 10],
    ):

        from pykeen.evaluation import RankBasedEvaluator

        self.evaluator = RankBasedEvaluator(ks=ks, filtered=True)
        self.path_manager = path_manager

        self.config = config.copy()
        self.config["pipeline"]["model"] = self.path_manager.model_name
        # self.optimizer = optiizer
        self.dataset = dataset(cache_root=self.path_manager.cache_directory)
        sub_graph_path = os.path.join(
            self.path_manager.cache_directory, "sub_graph.npz"
        )
        from pykeen.triples import TriplesFactory

        self.cid_smile = pd.read_csv(
            self.path_manager.smile_path, sep="\t", names=["cid", "smile"]
        )
        if not os.path.exists(sub_graph_path):
            build_sub_graph(
                self.dataset,
                sub_graph_path,
                skip_id=[
                    "PUBCHEM.COMPOUND:" + str(cid)
                    for cid in self.cid_smile[self.cid_smile["smile"].isna()][
                        "cid"
                    ].tolist()
                ],
                split=0.2,
            )
        loaded = np.load(sub_graph_path)
        my_training = loaded["training"]
        my_validation = loaded["validation"]
        my_testing = loaded["testing"]
        self.training = TriplesFactory.from_labeled_triples(my_training)
        self.validation = TriplesFactory.from_labeled_triples(my_validation)
        self.testing = TriplesFactory.from_labeled_triples(my_testing)
        # don't use new_with_restriction, as the masked entity are still in the entity_to_id list,
        # that triple with masked entity would still be sampled in negative samping,
        # The existence of these masked entity might excert a repealing force on every other entity, included chosen one

        self.results = dict()
        self.models = dict()

    def sanity_check(self):

        # check model
        for model in self.models:
            print("model : " + model)
            print("The model should have equal number of entity with the training set:")
            assert len(self.training.entity_to_id) == len(
                self.models[model].entity_embeddings()
            )
            print("checked")
            print(
                "\nThe model should have equal number of relation with the training set:"
            )
            assert len(self.training.relation_to_id) == len(
                self.models[model].relation_embeddings()
            )
            print("checked")

    # def save_full_graph(self):
    #     self.save("full_graph")
    #     return self

    # def save_sub_graph(self):
    #     self.save("sub_graph")
    #     return self

    # def save(self, model_key, trial):
    #     import torch
    #     import pickle

    #     torch.save(
    #         self.models[model_key],
    #         self.path_manager.model_path(model_key),
    #         pickle_protocol=pickle.HIGHEST_PROTOCOL,
    #     )
    #     return self

    # def save_all(self):
    #     for graph in self.models:
    #         self.save(graph)
    #     return self

    def load(self, graph, trial=None, map_location=None):
        import torch
        import os

        if trial == None:
            import json

            f = open(
                os.path.join(
                    self.path_manager.model_path(graph), "pipeline_config.json"
                )
            )
            pipeline_config = json.load(f)
            trial = pipeline_config["metadata"]["best_trial_number"]

        model_path = self.path_manager.model_path(graph, trial)
        if os.path.exists(model_path):
            self.models[graph] = torch.load(
                model_path,
                map_location=map_location,
            )
        return self

    def train_full_graph(self):
        print("\ntrain on full graph")
        config = self.config.copy()
        config["pipeline"]["training"] = self.dataset.training
        config["pipeline"]["validation"] = self.dataset.validation
        config["pipeline"]["testing"] = self.dataset.testing
        config["pipeline"]["save_model_directory"] = self.path_manager.model_dir_path(
            "full_graph"
        )
        print(config)
        hpo_pipeline_result = hpo_pipeline_from_config(config)
        hpo_pipeline_result.save_to_directory(
            self.path_manager.model_dir_path("full_graph")
        )
        print(hpo_pipeline_result.study.best_params)
        print(hpo_pipeline_result.study.best_trial.number)
        self.models["full_graph"] = self.load(
            "full_graph", hpo_pipeline_result.study.best_trial.number
        )
        return self

    def evaluate_full_graph(self, batch_size=2048):
        print("\nevaluate on full graph")
        self.results["full_graph"] = self.evaluate(
            "full_graph", self.dataset.testing.mapped_triples, batch_size=batch_size
        )
        return self

    def train_sub_graph(self):
        print("\ntrain on sub graph")
        config = self.config.copy()
        config["pipeline"]["training"] = self.training
        config["pipeline"]["validation"] = self.validation
        config["pipeline"]["testing"] = self.testing
        config["pipeline"]["save_model_directory"] = self.path_manager.model_dir_path(
            "sub_graph"
        )
        print(config)
        hpo_pipeline_result = hpo_pipeline_from_config(config)
        hpo_pipeline_result.save_to_directory(
            self.path_manager.model_dir_path("sub_graph")
        )
        print(hpo_pipeline_result.study.best_params)
        print(hpo_pipeline_result.study.best_trial.number)
        self.models["sub_graph"] = self.load(
            "sub_graph", hpo_pipeline_result.study.best_trial.number
        )
        return self

    def evaluate_sub_graph(self, batch_size=2048):
        print("\nevaluate on sub graph")
        self.results["sub_graph"] = self.evaluate(
            "sub_graph", self.testing.mapped_triples, batch_size=batch_size
        )
        return self

    def build_complemented_model(
        self, model_builder, complete_function=None, name="com_random"
    ):
        import torch

        with torch.no_grad():
            complemented_model = model_builder(
                triples_factory=self.dataset.training
            ).cpu()
            for e in self.dataset.training.entity_to_id:
                if e in self.training.entity_to_id:
                    complemented_model.entity_embeddings._embeddings.weight[
                        self.dataset.training.entity_to_id[e]
                    ] = self.models["sub_graph"].entity_embeddings()[
                        self.training.entity_to_id[e]
                    ]
                else:
                    if complete_function is None:
                        pass
                    else:
                        complemented_model.entity_embeddings._embeddings.weight[
                            self.dataset.training.entity_to_id[e]
                        ] = complete_function(e)
            for e in self.dataset.training.relation_to_id:
                if e in self.training.relation_to_id:
                    complemented_model.relation_embeddings._embeddings.weight[
                        self.dataset.training.relation_to_id[e]
                    ] = self.models["sub_graph"].relation_embeddings()[
                        self.training.relation_to_id[e]
                    ]
                else:
                    print("relation accidentally removed: " + str(e))
            self.models[name] = complemented_model
        return self

    def evaluate_complemented_model(self, name="com_random", batch_size=None):
        print("\nevaluate on complemented graph using " + name)
        self.results[name] = self.evaluate(
            name, self.dataset.testing.mapped_triples, batch_size=batch_size
        )
        return self

    def plot_results(self, plt_directory, mask=[]):
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime

        now = datetime.now()
        now_str = now.strftime("%m-%d-%Y_%H-%M-%S")
        selected_results = [result for result in self.results if result not in mask]
        metrics_dict = {
            "arithmetic_mean_rank": "Arithmetic Mean Rank",
            "geometric_mean_rank": "Geometric Mean Reciprocal Rank",
            "median_rank": "Median Rank",
            "harmonic_mean_rank": "Harmonic Mean Rank",
            "inverse_arithmetic_mean_rank": "Inverse Arithmetic Mean Rank",
            "inverse_geometric_mean_rank": "Inverse Geometric Mean Rank",
            "inverse_harmonic_mean_rank": "Inverse Harmonic Mean Rank",
            "inverse_median_rank": "Inverse Median Rank",
            "rank_std": "Rank Std",
            "rank_var": "Rank Var",
            "rank_mad": "Rank Mad",
            "adjusted_arithmetic_mean_rank": "Adjusted Arithmetic Mean Rank",
            "adjusted_arithmetic_mean_rank_index": "Adjusted Arithmetic Mean Rank Index",
        }
        for k in self.evaluator.ks:
            metrics_dict["hits_at_" + str(k)] = "hits_at_" + str(k)
        avg_only = [
            "adjusted_arithmetic_mean_rank",
            "adjusted_arithmetic_mean_rank_index",
        ]
        parts = ["head", "tail", "both"]
        labels = parts
        x = np.arange(len(labels))
        group_width = 0.35
        width = group_width / len(self.results)
        for metric in metrics_dict:
            fig, ax = plt.subplots()
            title = "Rank score of " + metrics_dict[metric]
            ax.set_title(title, loc="left")
            fig.set_size_inches(15, 7)
            for i, group in enumerate(selected_results):
                values = self.results[group].to_flat_dict()
                mark = x + i * group_width / len(selected_results) - group_width / 2
                retrive_string = ".realistic." + metric
                avg = ax.bar(
                    mark,
                    [values[index + retrive_string] for index in parts],
                    width,
                    label=group,
                    zorder=5,
                )
                if metric not in avg_only:
                    retrive_string = ".optimistic." + metric
                    max = ax.scatter(
                        mark,
                        [values[index + retrive_string] for index in parts],
                        c="limegreen",
                        marker="v",
                        zorder=10,
                    )
                    retrive_string = ".pessimistic." + metric
                    min = ax.scatter(
                        mark,
                        [values[index + retrive_string] for index in parts],
                        c="lightpink",
                        marker="^",
                        zorder=10,
                    )

            ax.legend(bbox_to_anchor=(1, 1.05), ncol=len(selected_results))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            import os

            plt.savefig(
                os.path.join(
                    plt_directory, now_str + "_" + title + "wo_" + "_".join(mask)
                )
            )
        return self

    def evaluate(self, name, mapped_triples, batch_size):
        import torch

        if torch.cuda.is_available():
            for model in self.models:
                self.models[model].cpu()
            torch.cuda.empty_cache()
            self.models[name].cuda()
        result = self.evaluator.evaluate(
            self.models[name], mapped_triples, batch_size=batch_size
        )
        return result

    def build_pandas(self):
        pubchem_compound = [
            entity_id
            for entity_id in self.dataset.entity_to_id
            if "PUBCHEM.COMPOUND" in entity_id
        ]
        selected_compound = [
            entity_id
            for entity_id in self.training.entity_to_id
            if "PUBCHEM.COMPOUND" in entity_id
        ]
        selected_cid = [entity_id.split(":")[-1] for entity_id in selected_compound]
        masked_compound = [
            entity_id
            for entity_id in set(self.dataset.entity_to_id).difference(
                set(self.training.entity_to_id.keys())
            )
        ]
        masked_cid = [entity_id.split(":")[-1] for entity_id in masked_compound]
        cid_embedding = pd.DataFrame.from_dict(
            {
                "cid": selected_cid,
                "embedding": [
                    self.models["sub_graph"]
                    .entity_embeddings()[self.training.entity_to_id[entity]]
                    .detach()
                    .cpu()
                    .numpy()
                    for entity in selected_compound
                ],
            }
        )
        self.cid_smile.cid = self.cid_smile.cid.astype(str)
        masked_cid_smile = self.cid_smile[self.cid_smile.cid.isin(masked_cid)]
        selected_cid_smile = self.cid_smile[self.cid_smile.cid.isin(selected_cid)]
        cid_embedding.cid = cid_embedding.cid.astype(selected_cid_smile.cid.dtypes)
        embedding_smile = selected_cid_smile.join(
            cid_embedding.set_index("cid"), on="cid"
        )
        print("sanity check")
        print("Check the mask and kg")
        print("masked compound and selected compound should not share any element:")
        assert bool(set(masked_compound) & set(selected_compound)) == False
        print("checked")
        print("\nmasked compound and selected compound should added to a full set:")
        assert (
            bool(set(pubchem_compound) == set(masked_compound + selected_compound))
            == True
        )
        print("checked")
        print(
            "\nsum of length of masked compound and selected compound should equal to the length of full set:"
        )
        assert len(pubchem_compound) == len(masked_compound) + len(selected_compound)
        print("checked")  # violated if masked_compound have duplication
        print(
            "\nlength of unique selected_compound should equal to the length of 80% of the full set:"
        )
        assert len(np.unique(selected_compound)) == len(pubchem_compound) - int(
            len(pubchem_compound) * 0.2
        )
        print("checked")  # violated if masked_compound have duplication
        print(
            "\nlength of unique masked_compound should equal to the length of 20% of the full set:"
        )
        assert len(np.unique(masked_compound)) == int(len(pubchem_compound) * 0.2)
        print("checked")  # violated if masked_compound have duplication
        print("\nthe selected set should be a (proper) subset of training set:")
        assert (
            bool(
                set(selected_compound).issubset(set(self.training.entity_to_id.keys()))
            )
            == True
        )
        print("checked")
        print("\nselected cid and selected compound should have same length:")
        assert len(selected_cid) == len(selected_compound)
        print("checked")
        print("\nis the selected set a subset of testing set?")
        print(
            bool(set(selected_compound).issubset(set(self.testing.entity_to_id.keys())))
        )
        print("Check the kg and kge")
        print("The model should have equal number of entity with the training set:")
        assert len(self.training.entity_to_id) == len(
            self.models["sub_graph"].entity_embeddings()
        )
        print("checked")
        print("\nThe model should have equal number of relation with the training set:")
        assert len(self.training.relation_to_id) == len(
            self.models["sub_graph"].relation_embeddings()
        )
        print("checked")

        print("Check the embedding_smile df")
        print("selected cid and embedding_smile should share same element:")
        assert bool(set(selected_cid) == set(embedding_smile.cid.astype(str))) == True
        print("checked")
        print(
            "\nlength of embedding_smile with embedding should equal to length os selected compound:"
        )
        assert embedding_smile.embedding.count() == len(selected_compound)
        print("checked")
        print(
            "\nmasked cid and embedding_smile with embedding should added to a full set:"
        )
        assert embedding_smile.embedding.count() + len(masked_compound) == len(
            pubchem_compound
        )
        print("checked")
        print("each smile from either embedding_smile or cid_smile should be equal:")
        for cid in embedding_smile.cid:
            assert embedding_smile.smile[embedding_smile.cid == cid].equals(
                self.cid_smile.smile[self.cid_smile.cid == cid]
            )
        print("checked")
        print(
            "each embedding from either embedding_smile or sub_graph_model should be equal:"
        )
        for cid in embedding_smile.cid:
            assert np.array_equal(
                embedding_smile.embedding[embedding_smile.cid == cid].iloc[0],
                self.models["sub_graph"]
                .entity_embeddings()[
                    self.training.entity_to_id["PUBCHEM.COMPOUND:" + str(cid)]
                ]
                .detach()
                .cpu()
                .numpy(),
            )
        print("checked")

        print("Check the masked_cid_smile df")
        print("masked cid and masked_cid_smile should share same element:")
        assert bool(set(masked_cid) == set(masked_cid_smile.cid.astype(str))) == True
        print("checked")
        embedding_smile_path, masked_cid_smile_path = self.path_manager.df_path()
        embedding_smile.to_pickle(embedding_smile_path)
        masked_cid_smile.to_pickle(masked_cid_smile_path)


def build_sub_graph(dataset, sub_graph_path, skip_id, split=0.2):
    compound = [
        entity_id
        for entity_id in dataset.entity_to_id
        if "PUBCHEM.COMPOUND" in entity_id and entity_id
    ]
    pubchem_compound = [
        entity_id
        for entity_id in dataset.entity_to_id
        if "PUBCHEM.COMPOUND" in entity_id and entity_id not in skip_id
    ]
    np.random.seed(1234)
    num_compound = len(compound)
    num_remove = int(num_compound * split)
    num_non_compound = len(dataset.entity_to_id) - num_compound
    num_relation = len(dataset.relation_to_id)
    random_compound = np.random.choice(
        pubchem_compound, replace=False, size=len(pubchem_compound)
    )
    random_compound_id = [
        dataset.training.entity_to_id[name] for name in random_compound
    ]
    my_training = dataset.training.mapped_triples.numpy()
    my_validation = dataset.validation.mapped_triples.numpy()
    my_testing = dataset.testing.mapped_triples.numpy()
    span_index = 0
    span = 1
    offset = 0
    my_entity = 0
    remaining_entity = dataset.entity_to_id
    while num_remove > len(dataset.entity_to_id) - len(remaining_entity):
        while True:
            candidate = random_compound_id[offset : offset + span]
            temp_training = my_training
            temp_validation = my_validation
            temp_testing = my_testing
            training_mask = np.logical_not(
                np.logical_or(
                    np.isin(temp_training[:, 0], candidate),
                    np.isin(temp_training[:, 2], candidate),
                )
            )
            temp_training = temp_training[training_mask]
            validation_mask = np.logical_not(
                np.logical_or(
                    np.isin(temp_validation[:, 0], candidate),
                    np.isin(temp_validation[:, 2], candidate),
                )
            )
            temp_validation = temp_validation[validation_mask]
            testing_mask = np.logical_not(
                np.logical_or(
                    np.isin(temp_testing[:, 0], candidate),
                    np.isin(temp_testing[:, 2], candidate),
                )
            )
            temp_testing = temp_testing[testing_mask]
            remaining_relation = np.unique(temp_training[:, 1])
            remaining_entity = reduce(
                np.union1d,
                (np.unique(temp_training[:, 0]), np.unique(temp_training[:, 2])),
            )
            if (
                len(remaining_relation) == num_relation
                and len(remaining_entity)
                == len(dataset.entity_to_id) - my_entity - span
            ):
                break
            else:
                if span == 1:
                    offset = offset + span
                span_index = 0
                span = 2 ** span_index
        offset = offset + span
        my_entity = my_entity + span
        my_training = temp_training
        my_validation = temp_validation
        my_testing = temp_testing
        span_index = span_index + 1
        span = 2 ** span_index
        if len(dataset.entity_to_id) - len(remaining_entity) + span > num_remove:
            span = num_remove - len(dataset.entity_to_id) + len(remaining_entity)
        if span == 0:
            break
    my_training = np.array(
        [
            [
                dataset.training.entity_id_to_label[row[0]],
                dataset.training.relation_id_to_label[row[1]],
                dataset.training.entity_id_to_label[row[2]],
            ]
            for row in my_training
        ]
    )
    my_validation = np.array(
        [
            [
                dataset.validation.entity_id_to_label[row[0]],
                dataset.validation.relation_id_to_label[row[1]],
                dataset.validation.entity_id_to_label[row[2]],
            ]
            for row in my_validation
        ]
    )
    my_testing = np.array(
        [
            [
                dataset.testing.entity_id_to_label[row[0]],
                dataset.testing.relation_id_to_label[row[1]],
                dataset.testing.entity_id_to_label[row[2]],
            ]
            for row in my_testing
        ]
    )
    print("sanity check")
    print("\nNo relation is accidentally removed:")
    assert len(np.unique(my_training[:, 1])) == num_relation
    print("checked")
    print("\nExactly " + str(num_remove) + " entity is removed")
    assert (
        len(np.union1d(np.unique(my_training[:, 0]), np.unique(my_training[:, 2])))
        == len(dataset.entity_to_id) - num_remove
    )
    print("checked")
    print("\nthe selected set should be a (proper) subset of training set:")
    assert (
        bool(
            set(
                np.union1d(np.unique(my_training[:, 0]), np.unique(my_training[:, 2]))
            ).issubset(set(dataset.training.entity_to_id.keys()))
        )
        == True
    )
    print("checked")
    my_entity = set(
        np.union1d(np.unique(my_training[:, 0]), np.unique(my_training[:, 2]))
    )
    print("\nRemoved entity are all compound")
    for removed in set(dataset.entity_to_id).difference(my_entity):
        assert "PUBCHEM.COMPOUND" in removed
    print("checked")
    np.savez_compressed(
        sub_graph_path,
        training=my_training,
        testing=my_testing,
        validation=my_validation,
    )
