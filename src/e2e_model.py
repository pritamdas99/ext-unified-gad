from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils import *
from predictor import *
from Pareto_fn import pareto_fn

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        # preds[probs[:,1] > thres] = 1
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

LABEL_DICT_KEYS = {
    'n':"node_label",
    'e':'edge_label',
    'g':'graph_label',
}

class UnifyMLPDetector(object):
    def __init__(self, total_nodes, dataset, dataloaders, cross_mode, args):
        self.args = args

        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]

        # the loss route
        input_route, output_route = cross_mode.split('2')
        self.input_route = [c for c in input_route] # ['n', 'e', 'g']
        self.output_route = [c for c in output_route] # ['n', 'e', 'g'] # the output of the model

        self.model = UNIMLP_E2E(
            total_nodes,
            in_feats=dataset.in_dim,
            embed_dims=args.hid_dim,
            stitch_mlp_layers=args.stitch_mlp_layers,
            final_mlp_layers=args.final_mlp_layers,
            dropout_rate=args.dropout,
            khop=args.khop,
            activation=args.act_ft,
            graph_batch_num=args.batch_size,
            n_heads=args.n_heads,
            n_layers_attention=args.n_layers_attention,
            ff_dim=args.ff_dim_transformer,
            output_route=output_route,
            input_route=input_route,
        ).to(args.device)

        self.loss_weight_dict = {}
        if 'n' in self.output_route:
            node_ab_count, node_total_count = sum([x.sum() for x in dataset.node_label]), sum(x.shape[0] for x in dataset.node_label)
            self.loss_weight_dict['n'] = (
                1/(node_ab_count/node_total_count),
                args.node_loss_weight 
            )
        if 'e' in self.output_route:
            edge_ab_count, edge_total_count = sum([x.sum() for x in dataset.edge_label]), sum(x.shape[0] for x in dataset.edge_label)
            self.loss_weight_dict['e'] = (
                1/(edge_ab_count/edge_total_count),
                args.edge_loss_weight 
            )
        if 'g' in self.output_route:
            graph_ab_count, graph_total_count = dataset.graph_label.sum(), dataset.graph_label.shape[0]
            self.loss_weight_dict['g'] = (
                1/(graph_ab_count/graph_total_count),
                args.graph_loss_weight 
            )

        # masks for single graph
        if dataset.is_single_graph:
            # mask_dicts = {}
            # self.is_single_graph = True
            # if 'n' in self.output_route:
            #     mask_dicts['n'] = {
            #         'train': dataset.train_mask_node_cur,
            #         'val': dataset.val_mask_node_cur,
            #         'test': dataset.test_mask_node_cur
            #     }
            # if 'e' in self.output_route:
            #     mask_dicts['e'] = {
            #         'train': dataset.train_mask_edge_cur,
            #         'val': dataset.val_mask_edge_cur,
            #         'test': dataset.test_mask_edge_cur
            #     }
            # if 'g' in self.output_route:
            #     # single graph cannot be classified
            #     raise NotImplementedError
            
            # self.model.mask_dicts = mask_dicts
            self.model.single_graph = True

        self.best_score = -1
        self.patience_knt = 0
        


    def get_loss(self, logits=[], labels=[]):
        loss_t = []
        loss_dict_t = []

        
        for logits_dict, labels_dict in zip(logits, labels):
            loss_list = []
            w_list = []
            c_list = []
            loss_items_dict = {'n': 0, 'e': 0, 'g': 0}
            for o_r in logits_dict:
                partial_loss = F.cross_entropy(logits_dict[o_r], labels_dict[LABEL_DICT_KEYS[o_r]], weight=torch.tensor([1., self.loss_weight_dict[o_r][0]], device=self.args.device)) # sensitive
                if o_r in self.input_route:
                    # loss = partial_loss if loss is None else (loss + partial_loss * self.loss_weight_dict[o_r][1])
                    loss_list.append(partial_loss)
                    w_list.append(1.0/len(self.input_route)) # FIXME: default loss average
                    c_list.append(0.01)
                loss_items_dict[o_r] = partial_loss.item()

            # return loss_list, loss_items_dict
            
            new_w_list = pareto_fn(w_list, c_list, model=self.model, num_tasks=len(loss_list), loss_list=loss_list)
            loss = 0
            for i in range(len(w_list)):
                loss += new_w_list[i]*loss_list[i]
            loss_t.append(loss)
            loss_dict_t.append(loss_items_dict)
        
        return loss_t, loss_dict_t
    
    @torch.no_grad()
    def get_probs(self, logits=[]):
        probs = []
        for logits_dict in logits:
            probs_dict = {}
            for o_r in logits_dict:
                probs_dict[o_r] = logits_dict[o_r].softmax(1)[:, 1]
            probs.append(probs_dict)
        return probs

    @torch.no_grad()
    def _single_eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            # guard against NaN in probs from unstable training
            if np.isnan(probs).any():
                probs = np.nan_to_num(probs, nan=0.0)
            score['MacroF1'] = get_best_f1(labels, probs)[0]
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)

        return score
    
    @torch.no_grad()
    def eval(self, labels, probs):
        result = {}
        for k in self.output_route:
            result[k] = self._single_eval(labels[k], probs[k])
        return result

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_ft, weight_decay=self.args.l2_ft)
        for epoch in tqdm( range(self.args.epoch_ft) ):
            loss_items_total_train = {k:0 for k in self.output_route }
            total_loss_graph = 0
            total_loss_node = 0
            for batched_data in self.train_dataloader:
                batched_graph = batched_data['graphs']
                batched_labels_dict = batched_data['labels_dict']
                batched_khop_graph = batched_data['sp_matrices']


                # FIXME: device issue?
                batched_graph = [graph.to(self.args.device) for graph in batched_graph]
                for t,label_dict in enumerate(batched_labels_dict):
                    for k,v in label_dict.items():
                        batched_labels_dict[t][k] = v.to(self.args.device)
                batched_khop_graph = [graph.to(self.args.device) for graph in batched_khop_graph]

                self.model.train()
                logits_dict= self.model(batched_graph, batched_khop_graph)
                loss, loss_items = self.get_loss(logits_dict, labels=batched_labels_dict)

                loss = torch.stack(loss).mean() # average the loss of different tasks
                result = {k: sum(d[k] for d in loss_items) / len(loss_items) for k in loss_items[0]}

                print(f"[DEBUG AFTER LOSS train] loss={loss} loss_items={loss_items} logits_dict={logits_dict[:10]}")

                for k in loss_items_total_train:
                    loss_items_total_train[k] += result[k]
                optimizer.zero_grad()
                loss.backward()
                # clip gradients to prevent explosion from Pareto-weighted updates
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                # scheduler.step()
                # # The following code is used to record the memory usage
                # py_process = psutil.Process(os.getpid())
                # print(f"CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB")
                # print(f"GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")

                # clear GPU cache
                del batched_data
                del batched_graph
                del batched_labels_dict
                del batched_khop_graph
                del logits_dict
                del loss
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                labels_dict_val_mul = {k:[] for k in self.output_route }
                probs_dict_val_mul = {k:[] for k in self.output_route }
                loss_items_total_val = {k:0 for k in self.output_route }
                # eval loop
                for batched_data in self.val_dataloader:

                    batched_graph = batched_data['graphs']
                    batched_labels_dict = batched_data['labels_dict']
                    batched_khop_graph = batched_data['sp_matrices']
                    # FIXME: device issue?
                    batched_graph = [graph.to(self.args.device) for graph in batched_graph]
                    
                    for t,label_dict in enumerate(batched_labels_dict):
                        for k,v in label_dict.items():
                            batched_labels_dict[t][k] = v.to(self.args.device)
                    
                    for k in batched_labels_dict[0]:
                        labels_mul_t=[]
                        for t,label_dict in enumerate(batched_labels_dict):
                            if k[0] in self.output_route:
                                labels_mul_t.append(label_dict[k])
                        labels_dict_val_mul[k[0]].append(labels_mul_t.to(self.args.device))
                            
                    
                    # for t,label_dict in enumerate(batched_labels_dict):
                    #     labels_dict_val_mul_t = {k:[] for k in self.output_route }
                    #     for k,v in label_dict.items():
                    #         batched_labels_dict[t][k] = v.to(self.args.device)
                    #         if k[0] in self.output_route:
                    #             labels_dict_val_mul_t[k[0]].append(v)
                    #     labels_dict_val_mul.append(labels_dict_val_mul_t)
                    
                    batched_khop_graph = [graph.to(self.args.device) for graph in batched_khop_graph]
                    
                    self.model.eval()
                    with torch.no_grad():
                        logits_dict = self.model(batched_graph, batched_khop_graph)
                        _, loss_items = self.get_loss(logits_dict, labels=batched_labels_dict)
                        print(f"[DEBUG AFTER VALIDATION]logits doict {logits_dict[:70]}")
                        result = {k: sum(d[k] for d in loss_items) / len(loss_items) for k in loss_items[0]}
                        for k in loss_items_total_val:
                            loss_items_total_val[k] += result[k] # notice next block
                        # for k in self.output_route:
                        #     loss_items_total_val[k] += loss_items[k]

                        probs = self.get_probs(logits_dict)
                        for k in probs[0]:
                            probs_mul_t=[]
                            for t, prob_t in enumerate(probs):
                                probs_mul_t.append(prob_t[k])
                            probs_dict_val_mul[k].append(probs_mul_t.to(self.args.device))
                    
                    del batched_data
                    del batched_graph
                    del batched_labels_dict
                    del batched_khop_graph
                    del logits_dict
                    del probs
                with torch.no_grad():
                    for k in self.output_route:
                        print("***************************##################",labels_dict_val_mul[k][0].shape)
                        labels_dict_val_mul[k] = torch.cat([t for t in labels_dict_val_mul[k]], dim=1)
                        probs_dict_val_mul[k] = torch.cat([t for t in probs_dict_val_mul[k]], dim=1)
                    # get eval score
                    score_val = self.eval(labels_dict_val_mul, probs_dict_val_mul)
                    del labels_dict_val_mul
                    del probs_dict_val_mul
                    # average different scores
                    score_overall_val = 0
                    for k in self.output_route:
                        score_overall_val += score_val[k][self.args.metric]
                    score_overall_val /= len(self.output_route)
                log_loss(['Train', 'Val'], [loss_items_total_train, loss_items_total_val])
                del loss_items_total_train
                del loss_items_total_val

                # select the best on val set
                if score_overall_val > self.best_score or self.patience_knt > self.args.patience:
                    torch.cuda.empty_cache()
                    self.best_score = score_overall_val
                    self.patience_knt = 0
                    labels_dict_test_mul = []
                    probs_dict_test_mul = []
                    loss_items_total_test = {k:0 for k in self.output_route }
                    # eval loop
                    for batched_data in self.test_dataloader:
                        batched_graph = batched_data['graphs']
                        batched_labels_dict = batched_data['labels_dict']
                        batched_khop_graph = batched_data['sp_matrices']

                        batched_graph = [graph.to(self.args.device) for graph in batched_graph]
                        for t,label_dict in enumerate(batched_labels_dict):
                            for k,v in label_dict.items():
                                batched_labels_dict[t][k] = v.to(self.args.device)
                    
                        for k in batched_labels_dict[0]:
                            labels_mul_t=[]
                            for t,label_dict in enumerate(batched_labels_dict):
                                if k[0] in self.output_route:
                                    labels_mul_t.append(label_dict[k])
                            labels_dict_test_mul[k[0]].append(labels_mul_t)
                        batched_khop_graph = [graph.to(self.args.device) for graph in batched_khop_graph]
                        
                        self.model.eval()
                        # get test result
                        with torch.no_grad():
                            logits_dict = self.model(batched_graph, batched_khop_graph)
                            _, loss_items = self.get_loss(logits_dict, labels=batched_labels_dict)
                            result = {k: sum(d[k] for d in loss_items) / len(loss_items) for k in loss_items[0]}
                            for k in loss_items_total_test:
                                loss_items_total_test[k] += result[k]
                                
                            probs = self.get_probs(logits_dict)
                            for k in probs[0]:
                                probs_mul_t=[]
                                for t, prob_t in enumerate(probs):
                                    probs_mul_t.append(prob_t[k])
                                probs_dict_test_mul[k].append(probs_mul_t)
                            
                        
                        del batched_data
                        del batched_graph
                        del batched_labels_dict
                        del batched_khop_graph
                        del logits_dict
                        del probs
                    # clear GPU cache
                    for k in self.output_route:
                        labels_dict_test_mul[k] = torch.cat([t for t in labels_dict_test_mul[k]], dim=1)
                        probs_dict_test_mul[k] = torch.cat([t for t in probs_dict_test_mul[k]], dim=1)
                    # get test score
                    score_test = self.eval(labels_dict_test_mul, probs_dict_test_mul)
                    del labels_dict_test_mul
                    del probs_dict_test_mul
                    torch.cuda.empty_cache()
                    # log to stdin
                    print(f'Epoch {epoch}: {self.best_score}\n{pprint.pformat(score_test)}')
                    if self.patience_knt > self.args.patience:
                        print("No patience")
                        break
                else:
                    self.patience_knt += 1

        return score_test