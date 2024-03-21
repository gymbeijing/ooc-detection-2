from model.realFND import Policy, FakeNewsClassifier, DomainClassifier
from torch import optim
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.utils as utils
import os
from dataset.twitterCOMMsDataset import TwitterCOMMsDataset
import torch.utils.data as data
import random
import argparse
from tqdm import tqdm
from dataset.newsCLIPpingsDataset import get_dataloader_2

"""
python -m trainers.train_agentNews --few_shot_topic bbc,guardian --base_model blip-2
python -m trainers.train_agentNews --few_shot_topic usa_today,washington_post --base_model blip-2
"""


class REINFORCE:
    def __init__(self, model, F, D):
        self.model = model
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        self.F = F.cuda().eval()
        self.D = D.cuda().eval()
        self.alpha = 0.5
        self.beta = 0.5

    def select_action(self, state):
        probs = self.model(Variable(state).cuda())
        # action = probs.multinomial().data
        # prob = probs[:, action[0,0]].view(1, -1)
        # log_prob = prob.log()
        # entropy = - (probs*probs.log()).sum()
        action = probs.argmax(dim=0)
        prob = probs[action]

        return action, prob

    def update_parameters(self, rewards, prob, gamma=0.99):
        R = torch.zeros(1, 1).to(device)
        loss = 0
        for i in (range(len(rewards))):
            R = gamma * R + rewards[i]
        
        loss = (R * prob).log()
		
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

    def update_state(self, state, action, sigma=0.01):
        state_prime = state.clone()
        if action < 768:
            idx = action % 768
            state_prime[idx] += sigma
        else:
            idx = action % 768
            state_prime[idx] -= sigma

        return state_prime
    
    def get_reward(self, state, news_label, domain_label):
        f_output = self.F(state)
        f_pr = f_output[news_label]
        d_output = self.D(state)
        d_pr = d_output[domain_label]

        return self.alpha * f_pr - self.beta * d_pr


def train(rl_algorithm, T, state, news_label, domain_label):
    rewards = []
    for t in range(T):
        action, prob = rl_algorithm.select_action(state)
        state_prime = rl_algorithm.update_state(state, action)
        reward = rl_algorithm.get_reward(state_prime, news_label, domain_label)
        rewards.append(reward)
        # print(reward)
        rl_algorithm.update_parameters(rewards, prob)
        state = state_prime
    return rewards


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    p = argparse.ArgumentParser()
    p.add_argument("--few_shot_topic", type=str, required=True)
    p.add_argument("--base_model", type=str, required=True)
    args = p.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    F = FakeNewsClassifier()
    D = DomainClassifier(event_num=4)
    F.load_state_dict(torch.load(os.path.join('./', 'real_fnd_output', f'fake_news_classifier_{args.few_shot_topic}.ckpt')))
    D.load_state_dict(torch.load(os.path.join('./', 'real_fnd_output', f'domain_classifier_{args.few_shot_topic}.ckpt')))
    Pi = Policy(768 * 2)

    root_dir = '/import/network-temp/yimengg/data/'

    train_data, train_dataloader, train_len = get_dataloader_2(target_agency=args.few_shot_topic, shuffle=True, batch_size=256, phase='train')


    reinforce = REINFORCE(Pi, F, D)

    min_range = 0
    max_range = len(train_data) - 1
    random_numbers = [random.randint(min_range, max_range) for _ in range(2000)]
    for rand in tqdm(random_numbers):
        episode = train_data[rand]["multimodal_emb"].to(device)
        news_label = train_data[rand]["label"]
        domain_label = train_data[rand]["domain_id"]
        rewards = train(reinforce, 20, episode, news_label, domain_label)

    torch.save(Pi.state_dict(), os.path.join('real_fnd_output', f'policy_network_{args.few_shot_topic}.ckpt'))
    
