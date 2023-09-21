import torch

def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics

def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]