import os
import yaml
import argparse
from tqdm import tqdm
from itertools import product
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from dataset import Datasets
from metrics import test
from model.AGFN import AGFN
from utils import CSV_LOG, cprint, init_best_metrics, log_metrics, makedir, set_seed







def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="which dataset to use, options: NetEase, Youshu, iFashion")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    args = parser.parse_args()

    return args


def main():
    set_seed()
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]


    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    dataset = Datasets(conf)
    conf["info"] = paras["info"]
    conf["gpu"] = paras["gpu"]
    conf["model"] = "AGFN_reproduce"
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    device = "cuda:"+ conf["gpu"]
    conf["device"] = device
    print(conf)
    csv_log = CSV_LOG(conf)
    for lr, l2_reg,embedding_size, num_layers,drop,leaky in \
            product(conf['lrs'], conf['l2_regs'], conf["embedding_sizes"], conf["num_layerss"],conf["drop"],conf["leaky"]):
        ### Path
        log_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" %(conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" %(conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" %(conf["dataset"], conf["model"])
        makedir(run_path)
        makedir(log_path)
        makedir(checkpoint_conf_path)
        makedir(checkpoint_model_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size
        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]
        settings += [str(conf['pre_epoch']), str(l2_reg),str(num_layers),str(drop),str(leaky),str(conf["gpu"])]
        title = f"{conf['model']}_{conf['dataset']}_L2_reg_{l2_reg}_num_layer_{num_layers}_drop_{drop}_leaky_{leaky}"
        setproctitle.setproctitle(title)
        cprint(title)
        conf["num_layers"] = num_layers

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting
            
        run = SummaryWriter(run_path)

        # model
        model = AGFN(dataset,embedding_size, num_layers,drop,conf["device"],leaky).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        pat = 0
        epoch = 0
        while epoch < conf['epochs']:
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i
                loss1,L2_loss = model(batch)
                loss = loss1 + l2_reg*L2_loss
                loss.backward()
                optimizer.step()
                loss_scalar = loss.detach()
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description("epoch: %d, loss: %.4f" %(epoch, loss_scalar))

                if epoch > conf["pre_epoch"] and (batch_anchor+1) % test_interval_bs == 0:  
                    metrics = {}
                    old_best = best_epoch
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path
                    , checkpoint_model_path, checkpoint_conf_path,
                     epoch, batch_anchor, best_metrics, best_perform, best_epoch)
                    cprint(f"The best epoch is {best_epoch}")
                    if old_best == best_epoch:
                        pat+= 1
                        if pat > conf["pat"]:
                            break
                    else:
                        pat = 0
                elif(batch_anchor+1) % int(batch_cnt * 50) == 0:
                    metrics = {}
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    for topk in [20,40,80]:
                        val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(epoch, topk, metrics["val"]["recall"][topk], metrics["val"]["ndcg"][topk])
                        test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(epoch, topk, metrics["test"]["recall"][topk], metrics["test"]["ndcg"][topk])
                        print(val_str)
                        print(test_str)
            epoch+=1
            if pat > conf["pat"]:
                csv_log.log_best_result(conf,l2_reg,num_layers,drop,leaky,best_epoch,best_metrics)
                break
            if not (epoch < conf['epochs']):
                csv_log.log_best_result(conf,l2_reg,num_layers,drop,leaky,best_epoch,best_metrics)


if __name__ == "__main__":
    main()