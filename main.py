
import os
import torch
from args import read_args
from tensorboardX import SummaryWriter
import logging
import torch.nn.functional as F
from collections import deque
from torch import optim
import time
from collections import defaultdict
from dataloader import train_generate
import random
from CIAN import CIAN  # ************************
import numpy as np
import json
torch.set_num_threads(1)



def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Model_Run(object):

    def __init__(self, arg):
        super(Model_Run, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        self.graph_path = args.datapath + '/path_graph.json'
        self.train_path = args.datapath + '/train_tasks_in_train.json'

        self.dev_path = args.datapath + '/dev_tasks.json'
        self.test_path = args.datapath + '/test_tasks.json'




        self.train_tasks = json.load(open(self.train_path))
        self.test_tasks = json.load(open(self.test_path))
        self.dev_tasks = json.load(open(self.dev_path))

        self.load_embed()


        args.embedding_size = self.embedding_size
        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols


        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(args.datapath + '/e1rel_e2_in_train.json'))

        self.ent2id = json.load(open(args.datapath + '/ent2ids'))
        self.id2ent = {id:ent for ent, id in self.ent2id.items()}
        self.num_ents = len(self.ent2id.keys())

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(args.datapath + '/rel2candidates_in_train.json'))

        logging.info('BUILDING CONNECTION MATRIX')
        degree = self.build_connection(max_=args.max_neighbor)

        if args.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        if args.test or args.random_embed:
            # gen symbol2id, without embedding
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()

        self.use_pretrain = use_pretrain



        self.CIAN = CIAN(args, self.num_symbols, self.embedding_size,
                                 self.symbol2vec, use_pretrain=self.use_pretrain, finetune=args.fine_tune)
        self.CIAN.to(args.device)

        self.batch_nums = 0

        self.ignored_parameters = list(map(id, self.CIAN.entity_encoder.symbol_emb.parameters()))
        self.base_params = filter(lambda p: id(p) not in self.ignored_parameters, self.CIAN.parameters())

        # self.optim = optim.Adam([
        #     {'params': self.base_params},
        # {'params': self.CIAN.entity_encoder.symbol_emb.parameters(), 'lr:': args.symbol_embed_lr}],
        #     lr=args.base_lr,  weight_decay=self.weight_decay
        # )

        self.parameters = filter(lambda p: p.requires_grad, self.CIAN.parameters())


        self.optim = optim.Adam(self.parameters, lr=args.lr, weight_decay=self.weight_decay)


        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.label_smoothing_eps = 0.2

        if args.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + args.prefix)

    def load_embed(self):
        # gen symbol2id, with embedding
        symbol_id = {}
        rel2id = json.load(open(args.datapath + '/relation2ids'))  # relation2id contains inverse rel
        ent2id = json.load(open(args.datapath + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if args.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(args.datapath + '/embed/entity2vec.' + args.embed_model)
            rel_embed = np.loadtxt(args.datapath + '/embed/relation2vec.' + args.embed_model)  # contain inverse edge

            if args.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            self.embedding_size = ent_embed.shape[1]

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key], :]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key], :]))

            #add unseen, new rels
            for rel, triples in self.train_tasks.items():
                heads_vec = []
                tails_vec = []
                for triple in triples:
                    heads_vec.append(list(ent_embed[ent2id[triple[0]], :]))
                    tails_vec.append(list(ent_embed[ent2id[triple[2]], :]))
                rel_vec = np.array(tails_vec).mean(axis=0) - np.array(heads_vec).mean(axis=0)
                if rel not in symbol_id.keys():
                    symbol_id[rel] = i
                    embeddings.append(list(rel_vec))
                    i += 1
                    symbol_id[rel + '_inv'] = i
                    embeddings.append(list(-1 * rel_vec))
                    i += 1

            for rel, triples in self.test_tasks.items():
                heads_vec = []
                tails_vec = []
                for triple in triples:
                    heads_vec.append(list(ent_embed[ent2id[triple[0]], :]))
                    tails_vec.append(list(ent_embed[ent2id[triple[2]], :]))
                rel_vec = np.array(tails_vec).mean(axis=0) - np.array(heads_vec).mean(axis=0)
                if rel not in symbol_id.keys():
                    symbol_id[rel] = i
                    embeddings.append(list(rel_vec))
                    i += 1
                    symbol_id[rel + '_inv'] = i
                    embeddings.append(list(-1 * rel_vec))
                    i += 1

            for rel, triples in self.dev_tasks.items():
                heads_vec = []
                tails_vec = []
                for triple in triples:
                    heads_vec.append(list(ent_embed[ent2id[triple[0]], :]))
                    tails_vec.append(list(ent_embed[ent2id[triple[2]], :]))
                rel_vec = np.array(tails_vec).mean(axis=0) - np.array(heads_vec).mean(axis=0)
                if rel not in symbol_id.keys():
                    symbol_id[rel] = i
                    embeddings.append(list(rel_vec))
                    i += 1
                    symbol_id[rel + '_inv'] = i
                    embeddings.append(list(-1 * rel_vec))
                    i += 1

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))

            embeddings = np.array(embeddings)
            self.symbol2id = symbol_id
            assert embeddings.shape[0] == len(symbol_id.keys())
            self.symbol2vec = embeddings

    def load_symbol2id(self):
        # only gen symbol2id
        symbol_id = {}
        rel2id = json.load(open(args.datapath + '/relation2ids'))  # relation2id contains inverse rel
        ent2id = json.load(open(args.datapath + '/ent2ids'))

        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

            #add unseen, new rels
        train_rels = list(self.train_tasks.keys())
        test_rels = list(self.test_tasks.keys())
        dev_rels = list(self.dev_tasks.keys())

        for rel in train_rels + test_rels + dev_rels:
            if rel not in symbol_id.keys():
                symbol_id[rel] = i
                i += 1
                symbol_id[rel + '_inv'] = i
                i += 1
        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(args.datapath + '/path_graph') as f:
            lines = f.readlines()
            for line in lines:
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))  # 1-n
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1]))  # n-1

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]  # rel
                self.connections[id_, idx, 1] = _[1]  # tail
        return degrees

    def get_degrees(self, nbrs_ent):
        degrees = []
        for sub_nbrs in nbrs_ent:
            sub_degrees = [self.e1_degrees[_] for _ in sub_nbrs]
            degrees.append(sub_degrees)
        degrees = torch.LongTensor(np.array(degrees)).to(args.device)
        return degrees


    def get_meta(self, left, right):
        left_connections = torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0)).to(args.device)
        left_degrees = torch.IntTensor([self.e1_degrees[_] for _ in left]).to(args.device)
        right_connections = torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0)).to(args.device)
        right_degrees = torch.IntTensor([self.e1_degrees[_] for _ in right]).to(args.device)


        return (left_connections, left_degrees, right_connections, right_degrees)




    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.CIAN.state_dict(), path)

    def load(self, path=None):
        if path:
            self.CIAN.load_state_dict(torch.load(path))

        else:
            self.CIAN.load_state_dict(torch.load(self.save_path))




    def train(self, args):


        logging.info('START TRAINING...')
        best_mrr = 0.0
        best_batches = 0

        losses = deque([], args.log_every)
        margins = deque([], args.log_every)
        for data in train_generate(args.datapath, args.batch_size, args.few, self.symbol2id, self.ent2id, self.e1rel_e2):


            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            self.batch_nums += 1

            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)


            support = torch.LongTensor(support).to(device=args.device)  # (few, 2)


            query = torch.LongTensor(query).to(device=args.device)  # (batch, 2)
            false = torch.LongTensor(false).to(device=args.device)  # (batch, 2)

            self.CIAN.train()

            positive_score, negative_score = self.CIAN(support, support_meta,
                                                           query, query_meta,
                                                           false, false_meta, is_train=True)


            margin_ = positive_score - negative_score
            loss = F.relu(self.margin - margin_).mean()
            margins.append(margin_.mean().item())
            lr = adjust_learning_rate(optimizer=self.optim, epoch=self.batch_nums, lr=args.lr,
                                      warm_up_step=args.warm_up_step,
                                      max_update_step=args.max_batches)

            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, args.grad_clip)
            self.optim.step()

            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                logging.info(
                    'Batch: {:d}, Avg_batch_loss: {:.6f}, lr: {:.6f}'.format(
                        self.batch_nums,
                        np.mean(losses),
                        lr))
                self.writer.add_scalar('Avg_batch_loss_every_log', np.mean(losses), self.batch_nums)

            if self.batch_nums % self.eval_every == 0:
                logging.info('Batch_nums is %d' % self.batch_nums)
                hits10, hits5, hits1, mrr = self.eval(mode='dev')
                self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                self.writer.add_scalar('HITS1', hits1, self.batch_nums)
                self.writer.add_scalar('MRR', mrr, self.batch_nums)
                self.save()

                if mrr > best_mrr:
                    self.save(self.save_path + '_best')
                    best_mrr = mrr
                    best_batches = self.batch_nums
                logging.info('Best_mrr is {:.6f}, when batch num is {:d}'.format(best_mrr, best_batches))

            if self.batch_nums == self.max_batches:
                self.save()
                break

            if self.batch_nums - best_batches > self.eval_every * 10:
                logging.info('Early stop!')
                self.save()
                break

    def eval(self, mode='dev'):

        with torch.no_grad():
            self.CIAN.eval()

            symbol2id = self.symbol2id
            few = self.few

            logging.info('EVALUATING ON %s DATA' % mode.upper())
            if mode == 'dev':
                test_tasks = json.load(open(args.datapath + '/dev_tasks.json'))
            else:
                test_tasks = json.load(open(args.datapath + '/test_tasks.json'))

            rel2candidates = self.rel2candidates

            hits10 = []
            hits5 = []
            hits1 = []
            mrr = []
            for query_ in test_tasks.keys():
                hits10_ = []
                hits5_ = []
                hits1_ = []
                mrr_ = []
                candidates = rel2candidates[query_]
                support_triples = test_tasks[query_][:few]
                support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]


                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

                support = torch.LongTensor(support_pairs).to(device=args.device)  # (few, 18)
                for triple in test_tasks[query_][few:]:
                    true = triple[2]
                    query_pairs = []
                    query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])
                    query_left = []
                    query_right = []
                    query_left.append(self.ent2id[triple[0]])
                    query_right.append(self.ent2id[triple[2]])

                    for ent in candidates:
                        if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                            query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])

                    query = torch.LongTensor(query_pairs).to(device=args.device)  # (few, 18)
                    query_meta = self.get_meta(query_left, query_right)

                    scores, _ = self.CIAN(support, support_meta,
                                              query, query_meta, false=None, false_meta=None, is_train=False)

                    scores.detach()
                    scores = scores.data

                    scores = scores.cpu().numpy()
                    sort = list(np.argsort(scores, kind='stable'))[::-1]
                    rank = sort.index(0) + 1
                    if rank <= 10:
                        hits10.append(1.0)
                        hits10_.append(1.0)
                    else:
                        hits10.append(0.0)
                        hits10_.append(0.0)
                    if rank <= 5:
                        hits5.append(1.0)
                        hits5_.append(1.0)
                    else:
                        hits5.append(0.0)
                        hits5_.append(0.0)
                    if rank <= 1:
                        hits1.append(1.0)
                        hits1_.append(1.0)
                    else:
                        hits1.append(0.0)
                        hits1_.append(0.0)
                    mrr.append(1.0 / rank)
                    mrr_.append(1.0 / rank)

                logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f}, MRR:{:.3f}'.format(query_,
                                                                                                   np.mean(
                                                                                                       hits10_),
                                                                                                   np.mean(hits5_),
                                                                                                   np.mean(hits1_),
                                                                                                   np.mean(mrr_),
                                                                                                   ))
                logging.info('Number of candidates: {}, number of test examples {}'.format(len(candidates), len(hits10_)))
            logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
            logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
            logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
            logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)


    def test_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for test')
        self.eval(mode='test')

    def eval_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for dev')
        self.eval(mode='test')


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':


    args = read_args()
    start_time = time.time()
    # setup random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    seed_everything(args.seed)

    # model execution
    model_run = Model_Run(args)

    # data analysis
    # model_run.data_analysis()

    # train/test model
    if args.test:
        print('*******************test*************************')
        model_run.test_()
    else:
        model_run.train(args)
        print('best checkpoint!')
        model_run.test_(args.save_path + '_best')
        print('last checkpoint!')
        model_run.test_()


    print(f'Total runing time:{(time.time() - start_time) / 60} min')

