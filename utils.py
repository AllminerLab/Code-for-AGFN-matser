import csv
from datetime import datetime
import random
import torch
import os
import numpy as np
import json
def cprint(words):
    print(f"\033[0;30;43m{words}\033[0m")

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def set_seed():
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform

class CSV_LOG():
    def __init__(self,conf):
        csv_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        self.csv_path = csv_path + "/res.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path,'w') as f:
                csv_write = csv.writer(f)
                csv_head = ["dataset","epoch","num_layer","drop","L2_nrom","leaky"]
                for metric in ["Recall@","NDCG@"]:
                    for k in conf["topk"]:
                        csv_head.append(metric+str(k))
                csv_write.writerow(csv_head)
    def log_best_result(self,conf,l2_reg,num_layers,drop,leaky,best_epoch,best_metrics):
        with open(self.csv_path,"a+") as csvfile: 
            writer = csv.writer(csvfile)
            row =[conf["dataset"],best_epoch,num_layers,drop,l2_reg,leaky]
            for metric in ["recall","ndcg"]:
                for k in conf["topk"]:
                    row.append(best_metrics["test"][metric][k])
            writer.writerow(row)
