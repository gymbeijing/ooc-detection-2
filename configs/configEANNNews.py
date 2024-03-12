import argparse


class ConfigEANNNews(object):
    def __init__(self):
        parser, args = parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.args.in_dim = 768
        self.args.class_num = 2


def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--batch_size", type=int, required=True, help="batch size")
    p.add_argument("--event_num", type=int, required=True, help="number of events")
    p.add_argument("--max_epochs", type=int, required=True, help="number of training epochs")
    p.add_argument("--hidden_dim", type=int, required=True, help="hidden_dim")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--target_agency", type=str, required=True)
    p.add_argument("--threshold", type=float, required=False, default=0.5,
                   help="threshold value for making the class prediction")

    args = p.parse_args()
    return p, args


# Unit test
if __name__ == '__main__':
    cfg = ConfigEANNNews()
    print(cfg.args)