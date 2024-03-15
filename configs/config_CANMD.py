import argparse


EXPERIMENT_ROOT_FOLDER = '.'


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=42, type=int,
                    help="Random seed for initialization")

# Model settings
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the model checkpoints and predictions will be written")
parser.add_argument("--base_model", default="blip-2", type=str)
parser.add_argument("--few_shot_topic", type=str)



# Optimization settings
parser.add_argument("--batch_size", default=256, type=int, help="Batch size used for training")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The maximum norm for backward gradients")
parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform")

# Optimization settings
parser.add_argument("--alpha", default=0.01, type=float, help="Alpha for reversal gradient layer in DAT and CDA")
parser.add_argument("--conf_threshold", default=0.6, type=float, help="Alpha for reversal gradient layer in DAT and CDA")


args = parser.parse_args()
args.classifier_dropout = 0.2
args.hidden_size = 768
args.hidden_dropout_prob = 768
args.num_labels = 2