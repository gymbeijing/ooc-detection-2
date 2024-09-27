import argparse


class ConfigConDA(object):
    def __init__(self):
        parser, args = parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.args.in_dim = 768
        self.args.proj_dim = 500   # original: 300
        self.args.hidden_size = 500   # original: 768 (for h->cls), 500 is for z->cls
        self.args.num_labels = 2
        self.args.learning_rate = 2e-4   # original: 2e-5
        self.args.model_save_path = "./saved_model"
        self.args.model_save_name = "ConDA_Cv.pt"
        self.args.classifier_dropout = 0.2


def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--batch_size", type=int, required=True, help="batch size")
    p.add_argument("--max_epochs", type=int, required=True, help="number of training epochs")
    p.add_argument("--tgt_topic", type=str, required=True, help="target topic")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--lambda_w", type=float, required=False, default=0.5,
                   help="weight of the contrastive loss")
    p.add_argument("--loss_type", type=str, required=True, help="simclr")
    p.add_argument("--lambda_mmd", type=float, required=False, default=1.0,
                   help="weight of the MMD")

    args = p.parse_args()
    return p, args


# Unit test
if __name__ == '__main__':
    cfg = ConfigConDA()
    print(cfg.args)