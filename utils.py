import random
from argparse import ArgumentParser
import torch
import sys

def add_argument():
    parser = ArgumentParser(description='Fusion Net QA')

    # system
    parser.add_argument('-log-file', type=str, dest='log_file', default='output.log')
    parser.add_argument('-log_per_updates', type=int, default=500)

    # Data Option
    parser.add_argument('-baidu-file', type=str, dest="baidu_file", default="data/baidu_data.json")
    parser.add_argument('-baidu-data', type=str, dest="baidu_data", default="data/baidu_data.pt")
    parser.add_argument('-train-file', type=str, dest="train_file", default="data/sogou_shuffle_train.json")
    parser.add_argument('-train-data', type=str, dest="train_data",
                        default="data/sogou_shuffle_train.pt")
    parser.add_argument('-valid-file', type=str, dest="valid_file", default="data/sogou_shuffle_valid.json")
    parser.add_argument('-valid-data', type=str, dest="valid_data",
                        default="data/sogou_shuffle_valid.pt")
    parser.add_argument('-embed_file', type=str, default='data/word_emb.pkl')
    parser.add_argument('-topk', type=int, dest="topk", default=30000)
    parser.add_argument('-dict', type=str, dest="dict_file", default='data/vocab.pt')

    # Train Option
    parser.add_argument('-epoch', type=int, dest="epoch", default=50)
    parser.add_argument('-batch', type=int, dest="batch", default=32)
    parser.add_argument('-device', type=int, dest="device", default=-1)
    parser.add_argument('-seed', type=int, dest="seed", default=1993)
    parser.add_argument('-exp-name', type=str, dest="exp_name", default=None, help="save model to model/$exp-name$/")
    parser.add_argument('-debug', dest="debug", action='store_true')
    parser.add_argument('-resume_snapshot', type=str, dest='resume_snapshot', default=None)
    parser.add_argument('-multi-gpu', action='store_true', dest='multi_gpu')

    # Model Option
    parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
    parser.add_argument('-pos-vec-size', type=int, dest="pos_vec_size", default=5)
    parser.add_argument('-ner-vec-size', type=int, dest="ner_vec_size", default=5)
    parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=128)
    parser.add_argument('-num-layers', type=int, dest='num_layers', default=2)
    parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0.3)
    parser.add_argument('-dropout-emb', type=float, dest='dropout_emb', default=0.3)
    parser.add_argument('-dropout', type=float, dest='dropout', default=0.3)
    parser.add_argument('-brnn', action='store_true', dest='brnn')
    parser.add_argument('-word-vectors', type=str, dest="word_vectors",
                        default='data/penny.cbow.dim300.bin')
    parser.add_argument('-embedding-cache', type=str, dest="embedding_cache",
                        default='data/embedding_cache.pt')
    parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='lstm', choices=["rnn", "gru", "lstm"])
    parser.add_argument('-multi-layer', type=str, dest='multi_layer_hidden', default='last',
                        choices=["concatenate", "last"])

    # Optimizer Option
    parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
    parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adamax")
    parser.add_argument('-lr', type=float, dest="lr", default=0.02)
    parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')
    parser.add_argument('-regular', type=float, default=0, dest="regular_weight", help='regular weight')

    parser.add_argument('-lr_adam_ce', type=float, default=2e-3)
    parser.add_argument('-lr_adam_rl', type=float, default=2e-3)
    parser.add_argument('-tune_top_embed', type=int, default=1000, help='finetune top-x embeddings.')
    parser.add_argument('-fix_embed', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('-use_qemb', action='store_true', dest='use_qemb')
    parser.add_argument('-feature_num', type=int, default=10)

    # Predict Option
    parser.add_argument('-model', nargs='+', type=str, dest="model_file", default=None)
    parser.add_argument('-test', type=str, dest="test_file", default='data/sogou_shuffle_valid.json')
    parser.add_argument('-output', type=str, dest="out_file", default='output/result')
    parser.add_argument('-question', action='store_true', dest="question")

    args = parser.parse_args()
    return args

def get_folder_prefix(args, model):
    import os
    if args.exp_name is not None:
        model_folder = 'saved_checkpoint' + os.sep + args.exp_name
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_prefix = model_folder + os.sep + args.exp_name
        with open(model_prefix + '.config', 'w') as output:
            output.write(model.__repr__())
            output.write(args.__repr__())
    else:
        model_folder = None
        model_prefix = None
    return model_folder, model_prefix

class AverageMeter(object):
    """Keep exponential weighted averages."""

    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)


class BatchGen:
    pos_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # sort by len
        data = sorted(data, key=lambda x: len(x[0]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            feature_len = len(batch[2][0][0])
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            question_len = max(len(x) for x in batch[3])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[3]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            # TODO question pos

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)

            # context_tokens = list()
            # for i, tokens in enumerate(batch[5]):
            #     context_tokens.append(tokens)

            context_tokens = batch[5]
            id_ = batch[-1]

            if not self.eval:
                y_s = torch.LongTensor(batch[6])
                y_e = torch.LongTensor(batch[7])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_mask,
                       question_id, question_mask, context_tokens, id_)
            else:
                yield (context_id, context_feature, context_tag, context_mask,
                       question_id, question_mask, context_tokens, y_s, y_e, id_)
