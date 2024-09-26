from dataset.newsCLIPpingsDatasetConDATriplet import get_dataloader
from configs.configConDANews import ConfigConDANews
from itertools import cycle
from tqdm import tqdm
import torch.distributed as dist
from model.mmd_code import MMD
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    cfg = ConfigConDANews()
    src_train_loader, src_train_dataset_size = get_dataloader(cfg, target_domain=['guardian', 'usa_today', 'bbc'], shuffle=True, phase="train")
    tgt_train_loader, tgt_train_dataset_size = get_dataloader(cfg, target_domain=['usa_today', 'washington_post'], shuffle=True, phase="train")

    if len(src_train_loader) == len(tgt_train_loader):
        double_loader = enumerate(zip(src_train_loader, tgt_train_loader))
    elif len(src_train_loader) < len(tgt_train_loader):
        print("Src smaller than Tgt")
        double_loader = enumerate(zip(cycle(src_train_loader), tgt_train_loader))   # zip() only iterates over the smallest iterable
    else:
        double_loader = enumerate(zip(src_train_loader, cycle(tgt_train_loader)))

    with tqdm(double_loader, disable=False and dist.get_rank() > 0) as loop:
        count = 0
        for i, (src_data, tgt_data) in loop:
            # (1) Prepare the data inputs and labels
            src_emb, src_perturb_emb, src_negative_emb, src_labels = src_data["original_multimodal_emb"], src_data["positive_multimodal_emb"], src_data["negative_multimodal_emb"], src_data["original_label"]
            batch_size = src_emb.shape[0]

            tgt_emb, tgt_perturb_emb, tgt_labels = tgt_data["original_multimodal_emb"], tgt_data["positive_multimodal_emb"], tgt_data["original_label"]
            src_emb, tgt_emb = src_emb.to(device), tgt_emb.to(device)
            mmd = MMD(src_emb, tgt_emb, kernel='rbf')
            print(f"MMD: {mmd}")
            count += 1
            if count > 10:
                break
            