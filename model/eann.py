from torch.autograd import Variable, Function
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F


class ReverseLayerF(Function):

    @staticmethod
    def forward(self, x):
        self.lambd = 1   # default: 1
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)
    

def grad_reverse(x):
    #return ReverseLayerF()(x)
	return ReverseLayerF.apply(x)


# Neural Network Model (1 hidden layer)
class EANN(nn.Module):
    def __init__(self, args):
        super(EANN, self).__init__()
        self.args = args

        self.event_num = args.event_num

        C = args.class_num
        self.hidden_size = args.hidden_dim

        self.dropout = nn.Dropout(0.2)

        # self.image_fc1 = nn.Linear(args.in_dim,  self.hidden_size)   # (vis-fc): 768 -> 768
        # self.text_fc1 = nn.Linear(args.in_dim,  self.hidden_size)   # (txt-fc): 256 -> 256
        self.image_text_fc1 = nn.Linear(args.in_dim, self.hidden_size)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))   # (pred-fc): 768 -> 2
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))   # (adv-fc1): 768->768
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))   # (adv-fc2):  256 -> 2
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))
    

    def forward(self, text_image):
        """
        text: [bs, 256]
        image: [bs, 256]
        """

        # image = F.leaky_relu(self.image_fc1(image))   # (vis-fc)
        # text = F.leaky_relu(self.text_fc1(text))   # replace Text-CNN with linear layer (text-fc)
        # text_image = torch.cat((text, image), 1)   # [512, 1]?

        text_image = F.leaky_relu(self.image_text_fc1(text_image))

        ### Fake or real
        class_output = self.class_classifier(text_image)
        ## Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
    

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)