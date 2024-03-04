"""
(venv_py38) python -m trainers.train_conDATripletNews --batch_size 256 --max_epochs 20 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
"""
import logging
import os
import subprocess
from itertools import count
from multiprocessing import Process
from model.conDA import ProjectionMLP, MLLMClassificationHead, ContrastiveLearningLossZModule

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
# from transformers import *
from itertools import cycle
from functools import reduce

from dataset.newsCLIPpingsDatasetConDATriplet import get_dataloader
from configs.configConDANews import ConfigConDANews
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0,os.getcwd())   # inserts the current working directory at the beginning of the search path
torch.manual_seed(int(1001))

DISTRIBUTED_FLAG = False


# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


def setup_distributed(port=29500):
    if not DISTRIBUTED_FLAG:
        return 0, 1   # indicating that distributed training is not configured

    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1   # If it's not feasible, it returns default values (0, 1).

    # if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
    #     from mpi4py import MPI   # cannot be installed
    #     mpi_rank = MPI.COMM_WORLD.Get_rank()
    #     mpi_size = MPI.COMM_WORLD.Get_size()
    #
    #     os.environ["MASTER_ADDR"] = '127.0.0.1'
    #     os.environ["MASTER_PORT"] = str(port)
    #
    #     dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
    #     return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def distributed():
    # return dist.is_available() and dist.is_initialized()
    return False ## only because I want to use one GPU


def summary(model: nn.Module, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:   # logits: [bs, 2], labels: [bs, ]
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:   # logits: [bs,]
        classification = (logits > 0).long().flatten()   # ?
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def train(model: nn.Module, optimizer, device: str, src_loader: DataLoader,
          tgt_loader: DataLoader, summary_writer: SummaryWriter, desc='Train', lambda_w=0.5):
    model.train()

    src_train_accuracy = 0
    tgt_train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    if len(src_loader) == len(tgt_loader):
        double_loader = enumerate(zip(src_loader, tgt_loader))
    elif len(src_loader) < len(tgt_loader):
        print("Src smaller than Tgt")
        double_loader = enumerate(zip(cycle(src_loader), tgt_loader))   # zip() only iterates over the smallest iterable
    else:
        double_loader = enumerate(zip(src_loader, cycle(tgt_loader)))
    with tqdm(double_loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        torch.cuda.empty_cache()
        for i, (src_data, tgt_data) in loop:
            # (1) Prepare the data inputs and labels
            src_emb, src_perturb_emb, src_negative_emb, src_labels = src_data["original_multimodal_emb"], src_data["positive_multimodal_emb"], src_data["negative_multimodal_emb"], src_data["original_label"]
            src_emb, src_perturb_emb, src_negative_emb, src_labels = src_emb.to(device), src_perturb_emb.to(device), src_negative_emb.to(device), src_labels.to(device)
            batch_size = src_emb.shape[0]

            tgt_emb, tgt_perturb_emb, tgt_labels = tgt_data["original_multimodal_emb"], tgt_data["positive_multimodal_emb"], tgt_data["original_label"]
            tgt_emb, tgt_perturb_emb, tgt_labels = tgt_emb.to(device), tgt_perturb_emb.to(device), tgt_labels.to(device)

            # (2) optimizer set to zero_grad()
            optimizer.zero_grad()

            # (3) model is the overall model, including MLLMClsHead and the projection mlp,
            # model.forward() will address the loss computation on each module
            output_dic = model(src_emb, src_perturb_emb, src_negative_emb, tgt_emb, tgt_perturb_emb, src_labels, tgt_labels)

            loss = output_dic.total_loss

            # (4) Back-propagation: compute the gradients
            loss.backward()

            # (5) Update the model parameters
            optimizer.step()

            # (6) Evaluate
            src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
            src_train_accuracy += src_batch_accuracy
            tgt_batch_accuracy = accuracy_sum(output_dic.tgt_logits, tgt_labels)
            tgt_train_accuracy += tgt_batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), src_acc=src_train_accuracy / train_epoch_size,
                             tgt_acc=tgt_train_accuracy / train_epoch_size,
                             mmd=output_dic.mmd.item(), 
                             src_LCE_real=output_dic.src_ce_loss_real.item(),
                             src_LCE_perturb=output_dic.src_ce_loss_perturb.item(),
                            #  src_LCE_negative=output_dic.src_ce_loss_negative.item(),   #### newly added negative cross-entropy loss ######
                            #  src_ltriplet=output_dic.src_triplet_loss.item(),
                             src_lctr=output_dic.src_ctr_loss.item())

    return {
        "train/src_accuracy": src_train_accuracy,
        "train/tgt_accuracy": tgt_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []
            # print(example)
            for data in example:
                emb, labels = data["original_multimodal_emb"], data["original_label"]
                emb, labels = emb.to(device), labels.to(device)
                batch_size = emb.shape[0]

                ###### For the z instead of h input to the model ######
                z = model.mlp(emb)
                ###############

                # logits = model(emb)   # What is the model here? it's the mllm_cls_head
                logits = model.model(z)   # What is the model here? it's the entire ConDA, compatible with ContrastiveLearningAndTripletLossZModule
                # loss, softmax_logits = model.compute_loss(logits, labels=labels), model.compute_softmax_logits(logits)
                loss, softmax_logits = model.model.compute_loss(logits, labels=labels), model.model.compute_softmax_logits(logits)
                losses.append(loss)
                logit_votes.append(softmax_logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size


            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }


def test_time_adaptation(model, test_loader):
    model.train()  # Set model to train mode for adaptation
    for data in test_loader:
        emb, labels = data["original_multimodal_emb"], data["original_label"]
        emb, labels = emb.to(device), labels.to(device)

        # For ContrastiveLearningAndTripletLossModule, use:
        # outputs = model(emb) # Updating EMA of E[x] and Var[x]

        # For ContrastiveLearningAndTripletLossZModule, use:
        z = model.mlp(emb)
        _ = model.model(z)


def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        # torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()
    return output_d


# def run(cfg,
#         model_save_path,
#         model_save_name,
#         batch_size,
#         loss_type,
#         max_epochs=None,
#         device=None,
#         epoch_size=None,
#         seed=None,
#         token_dropout=None,
#         large=False,
#         learning_rate=2e-5,
#         weight_decay=0,
#         load_from_checkpoint=False,
#         lambda_w=0.5,
#         checkpoint_name='',
#         **kwargs):
def run(cfg, device):

    model_save_path = cfg.args.model_save_path
    model_save_name = cfg.args.model_save_name
    batch_size = cfg.args.batch_size
    loss_type = cfg.args.loss_type
    max_epochs = cfg.args.max_epochs
    epoch_size = None
    seed = None
    token_dropout = None
    large = False
    learning_rate = cfg.args.learning_rate
    weight_decay = 0
    load_from_checkpoint = False
    lambda_w = cfg.args.lambda_w
    checkpoint_name = ''

    args = locals()   # returns a dictionary containing the current local symbol table
    rank, world_size = setup_distributed()   # if not set to distributed, rank=0, world_size=1

    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    # if device=='cpu':
    #    print("Could not find GPU")
    #    exit()

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    # Set the logs directory
    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)

    # Creates a SummaryWriter object with the specified log directory (logdir).
    # The SummaryWriter is typically used for writing TensorBoard logs,
    # which can be visualized to monitor training metrics.
    writer = SummaryWriter(logdir) if rank == 0 else None

    import torch.distributed as dist
    if distributed() and rank > 0:
        # Synchronize processes before moving to the next stage
        dist.barrier()

    # model_name = 'roberta-large' if large else 'roberta-base'
    # tokenization_utils.logger.setLevel('ERROR')   # tokenization_utils defined at where?
    # tokenizer = RobertaTokenizer.from_pretrained('/home/abhatt43/projects/huggingface_repos/' + model_name)
    # roberta_model = RobertaForContrastiveClassification.from_pretrained(
    #     '/home/abhatt43/projects/huggingface_repos/' + model_name).to(device)
    # (1) classification MLP
    mllm_cls_head = MLLMClassificationHead(cfg).to(device)

    # (2) projection MLP
    mlp = ProjectionMLP(cfg).to(device)

    # (3) the entire contrastive learning framework
    model = ContrastiveLearningLossZModule(model=mllm_cls_head, mlp=mlp, loss_type=loss_type, logger=writer, device=device,
                                      lambda_w=lambda_w)
    # one process
    if rank == 0:
        summary(model)
        if distributed():   # always false
            dist.barrier()

    # more than one processes
    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)

    src_excluded_topic = cfg.args.target_domain.split(',')   # e.g. bbc
    tgt_excluded_topic = ['bbc', 'guardian', 'usa_today', 'washington_post']
    for topic in src_excluded_topic:
        tgt_excluded_topic.remove(topic)   # e.g. ['guardian', 'usa_today', 'washington_post']
    print(f"src_excluded_topic: {src_excluded_topic}")
    print(f"tgt_excluded_topic: {tgt_excluded_topic}")
    # loading data
    src_train_loader, src_train_dataset_size = get_dataloader(cfg, target_domain=src_excluded_topic, shuffle=True, phase="train")
    src_validation_loader, src_validation_dataset_size = get_dataloader(cfg, target_domain=src_excluded_topic, shuffle=False, phase="test")

    tgt_train_loader, tgt_train_dataset_size = get_dataloader(cfg, target_domain=tgt_excluded_topic, shuffle=True, phase="train")
    tgt_validation_loader, tgt_validation_dataset_size = get_dataloader(cfg, target_domain=tgt_excluded_topic, shuffle=False, phase="test")

    print(f"source train dataset size: {src_train_dataset_size}, target train dataset size: {tgt_train_dataset_size}")
    print(f"source validation dataset size: {src_validation_dataset_size}, target validation dataset size: {tgt_validation_dataset_size}")

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)   # count(1) is an infinite iterator

    best_validation_accuracy = 0
    without_progress = 0
    earlystop_epochs = 5

    for epoch in epoch_loop:

        if world_size > 1:
            src_train_loader.sampler.set_epoch(epoch)
            src_validation_loader.sampler.set_epoch(epoch)
            tgt_train_loader.sampler.set_epoch(epoch)
            tgt_validation_loader.sampler.set_epoch(epoch)

        train_metrics = train(model, optimizer, device, src_train_loader, tgt_train_loader, writer,
                              f'Epoch {epoch}', lambda_w=lambda_w)
        # validation_metrics = validate(mllm_cls_head, device,
        #                               src_validation_loader)  ## we are only using supervision on the source. Wrong using src_validation_loader!!!
        ## Test-time Adaptation ###
        # test_time_adaptation(mllm_cls_head, tgt_validation_loader)   # compatible with ContrastiveLearningAndTripletLossModule
        test_time_adaptation(model, tgt_validation_loader)   # compatible with ContrastiveLearningAndTripletLossZModule
        ###########################
        # validation_metrics = validate(mllm_cls_head, device,
        #                               tgt_validation_loader)  ## we are only using supervision on the source, compatible with ContrastiveLearningAndTripletLossModule
        validation_metrics = validate(model, device,
                                      tgt_validation_loader)  ## we are only using supervision on the source, compatible with ContrastiveLearningAndTripletLossZModule


        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        combined_metrics["train/src_accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        if rank == 0:
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                without_progress = 0
                best_validation_accuracy = combined_metrics["validation/accuracy"]

                model_to_save = mllm_cls_head
                torch.save(dict(
                    epoch=epoch,
                    model_state_dict=model_to_save.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    # args=args
                ),
                    os.path.join(model_save_path, model_save_name)
                )

        without_progress += 1

        if without_progress >= earlystop_epochs:
            break


def main(cfg, device):
    # number of process = number of gpus
    nproc = int(subprocess.check_output([sys.executable, '-c', "import torch;"
                                                               "print(torch.cuda.device_count() if torch.cuda.is_available() else 1)"]))
    nproc = 1
    # for machine compatibility

    if nproc > 1:
        print(f'Launching {nproc} processes ...', file=sys.stderr)

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(29500)
        os.environ['WORLD_SIZE'] = str(nproc)
        os.environ['OMP_NUM_THREAD'] = str(1)
        subprocesses = []

        for i in range(nproc):
            os.environ['RANK'] = str(i)
            os.environ['LOCAL_RANK'] = str(i)
            process = Process(target=run, kwargs=vars(cfg.args))
            process.start()
            subprocesses.append(process)

        for process in subprocesses:
            process.join()
    else:
        run(cfg, device)   # get a dictionary of the object's attributes, args is obtained from parse.parse_args()


if __name__ == '__main__':
    cfg = ConfigConDANews()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    main(cfg, device)

