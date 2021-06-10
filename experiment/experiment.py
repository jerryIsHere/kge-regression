class Experiment():
    def __init__(self, dataset, masked_id, sub_graph_path, model_initializer, cache_root, ks=[1, 3, 5, 10]):
        self.cache_root = cache_root

        if self.cache_root[-1] != '/':
            self.cache_root += '/'
        from pykeen.evaluation import RankBasedEvaluator
        self.evaluator = RankBasedEvaluator(ks=ks)
        from torch.optim import Adam
        self.optimizer = Adam
        self.dataset = dataset
        self.model_initializer = model_initializer
        from pykeen.triples import TriplesFactory
        import numpy as np
        import os
        if os.path.exists(sub_graph_path):
            loaded = np.load(sub_graph_path)
            my_training = loaded['training']
            my_testing = loaded['testing']
        else:
            my_training = [row for row in np.array(
                dataset.training.mapped_triples) if row[0] not in masked_id and row[2]not in masked_id]
            my_testing = [row for row in np.array(
                dataset.testing.mapped_triples) if row[0] not in masked_id and row[2]not in masked_id]
            my_training = np.array([[dataset.training.entity_id_to_label[row[0]],
                                     dataset.training.relation_id_to_label[row[1]],
                                     dataset.training.entity_id_to_label[row[2]]
                                     ]for row in my_training])
            my_testing = np.array([[dataset.testing.entity_id_to_label[row[0]],
                                    dataset.testing.relation_id_to_label[row[1]],
                                    dataset.testing.entity_id_to_label[row[2]]
                                    ]for row in my_testing])
            np.savez_compressed(
                sub_graph_path, training=my_training, testing=my_testing)
        my_training = TriplesFactory.from_labeled_triples(my_training)
        my_testing = TriplesFactory.from_labeled_triples(my_testing)
        # don't use new_with_restriction, as the masked entity are still in the entity_to_id list,
        # that triple with masked entity would still be sampled in negative samping,
        # The existence of these masked entity might excert a repealing force on every other entity, included chosen one
        self.training = my_training
        self.testing = my_testing
        self.results = dict()
        self.models = dict()

    def save(self, model_name):
        import torch
        import pickle
        for graph in self.models:
            torch.save(self.models[graph], self.cache_root + graph + '_' + model_name,
                       pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def load(self, graph, model_name, map_location=None):
        import torch
        import os
        if os.path.exists(self.cache_root + graph + '_' + model_name):
            self.models[graph] = torch.load(
                self.cache_root + graph + '_' + model_name, map_location=map_location)
        return self

    def train_full_graph(self, num_epochs=None, patience=2, frequency=10):
        print('\ntrain on full graph')
        model = self.model_initializer(triples_factory=self.dataset.training)

        from pykeen.training import SLCWATrainingLoop

        training_loop = SLCWATrainingLoop(
            model=model, optimizer=self.optimizer(params=model.get_grad_params()))
        training_loop.train(num_epochs=num_epochs)
        self.models['full_graph'] = model
        return self

    def evaluate_full_graph(self, batch_size=None):
        print('\nevaluate on full graph')
        self.results['full_graph'] = \
            self.evaluate(
                'full_graph', self.dataset.testing.mapped_triples, batch_size=batch_size)
        return self

    def train_sub_graph(self, num_epochs=None, patience=2, frequency=10):
        print('\ntrain on sub graph')
        my_model = self.model_initializer(triples_factory=self.training)

        from pykeen.training import SLCWATrainingLoop
        training_loop = SLCWATrainingLoop(
            model=my_model, optimizer=self.optimizer(params=my_model.get_grad_params()))
        training_loop.train(num_epochs=num_epochs)
        self.models['sub_graph'] = my_model
        return self

    def evaluate_sub_graph(self, batch_size=None):
        print('\nevaluate on sub graph')
        self.results['sub_graph'] = self.evaluate(
            'sub_graph', self.testing.mapped_triples, batch_size=batch_size)
        return self

    def build_complemented_model(self, complete_function=None, name='com_random'):
        import torch
        with torch.no_grad():
            complemented_model = self.model_initializer(
                triples_factory=self.dataset.training).cpu()
            for e in self.dataset.training.entity_to_id:
                if e in self.training.entity_to_id:
                    complemented_model.entity_embeddings._embeddings.weight[self.dataset.training.entity_to_id[e]]\
                        = self.models['sub_graph'].entity_embeddings()[self.training.entity_to_id[e]]
                else:
                    if complete_function is None:
                        pass
                    else:
                        complemented_model.entity_embeddings._embeddings.weight[self.dataset.training.entity_to_id[e]]\
                            = complete_function(e)
            for e in self.dataset.training.relation_to_id:
                if e in self.training.relation_to_id:
                    complemented_model.relation_embeddings._embeddings.weight[self.dataset.training.relation_to_id[e]]\
                        = self.models['sub_graph'].relation_embeddings()[self.training.relation_to_id[e]]
                else:
                    print('relation accidentally removed: '+str(e))
            self.models[name] = complemented_model
        return self

    def evaluate_complemented_model(self, name='com_random', batch_size=None):
        print('\nevaluate on complemented graph using ' + name)
        self.results[name] = self.evaluate(
            name, self.dataset.testing.mapped_triples, batch_size=batch_size)
        return self

    def plot_results(self, mask=[]):
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        selected_results = [
            result for result in self.results if result not in mask]
        metrics_dict = {
            'mean_rank': 'Mean Rank', 'mean_reciprocal_rank': 'Mean Reciprocal Rank',
            'adjusted_mean_rank': 'Adjusted Mean Rank',
        }
        for k in self.evaluator.ks:
            metrics_dict['hits_at_'+str(k)] = 'hits_at_'+str(k)
        avg_only = ['adjusted_mean_rank']
        parts = ['head', 'tail', 'both']
        labels = parts
        x = np.arange(len(labels))
        group_width = 0.35
        width = group_width/len(self.results)
        for metric in metrics_dict:
            fig, ax = plt.subplots()
            ax.set_title('Rank score of '+metrics_dict[metric], loc='left')
            fig.set_size_inches(15, 7)
            for i, group in enumerate(selected_results):
                values = self.results[group].to_flat_dict()
                mark = x + i*group_width/len(selected_results) - group_width/2
                retrive_string = '.avg.'+metric
                avg = ax.bar(mark, [values[index + retrive_string]
                                    for index in parts], width, label=group, zorder=5)
                if metric not in avg_only:
                    retrive_string = '.best.'+metric
                    max = ax.scatter(mark, [values[index + retrive_string] for index in parts],
                                     c='limegreen', marker='v', zorder=10)
                    retrive_string = '.worst.'+metric
                    min = ax.scatter(mark, [values[index + retrive_string] for index in parts],
                                     c='lightpink', marker='^', zorder=10)

            ax.legend(bbox_to_anchor=(1, 1.05), ncol=len(selected_results))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            plt.show()
        return self

    def evaluate(self, name, mapped_triples, batch_size):
        import torch
        if torch.cuda.is_available():
            for model in self.models:
                self.models[model].cpu()
            torch.cuda.empty_cache()
            self.models[name].cuda()
        result = self.evaluator.evaluate(
            self.models[name], mapped_triples, batch_size=batch_size)
        return result
