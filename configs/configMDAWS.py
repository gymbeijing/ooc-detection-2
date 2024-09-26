import argparse


class ConfigMDAWS(object):
    def __init__(self):
        parser, args = parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.args.batch_size = 256
        self.args.max_epochs = 10
        self.args.base_model = 'blip-2'
        self.args.hidden_size = 768
        self.args.domain_class = 4
        self.args.pre_train_epochs = 5
        self.args.is_weight_avg  = False


def parse():
    p = argparse.ArgumentParser()

    # p.add_argument("--batch_size", type=int, required=True, help="batch size")
    # p.add_argument("--max_epochs", type=int, required=True, help="number of training epochs")
    # p.add_argument("--hidden_dim", type=int, required=True, help="hidden_dim")
    # p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--few_shot_topic", type=str, required=True)   # for twitter-comms
    # p.add_argument("--target_agency", type=str, required=True)   # for newsclippings
    p.add_argument("--threshold", type=float, required=False, default=0.5,
                   help="threshold value for making the class prediction")

    args = p.parse_args()
    return p, args


# Unit test
if __name__ == '__main__':
    cfg = ConfigMDAWS()
    print(cfg.args)