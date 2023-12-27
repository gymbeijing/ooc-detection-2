import argparse


class ConfigTwoTasks(object):
    def __init__(self):
        parser, args = parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.in_dim = 768
        self.out_dim = 2


def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--batch_size", type=int, required=True, help="batch size")
    p.add_argument("--max_epochs", type=int, required=True, help="number of training epochs")
    p.add_argument("--few_shot_topic", type=str, required=False,
                       help="topic that will not be included in the training")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--threshold", type=float, required=False, default=0.5,
                       help="threshold value for making the class prediction")
    p.add_argument("--alpha", type=float, required=False, default=0.5,
                       help="weight assigned to the residual part")

    args = p.parse_args()
    return p, args