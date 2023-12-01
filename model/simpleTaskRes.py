from torch import nn
import torch
from lavis.models import load_model_and_preprocess


CUSTOM_TEMPLATES = {
    "Twitter-COMMs": "a piece of news in {}."   # {domain}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleTaskResLearner(nn.Module):
    def __init__(self, alpha, base_text_features):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("base_prompt_features", base_text_features)
        self.base_text_features = base_text_features
        # Learnable part
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        return self.base_text_features + self.alpha * self.text_feature_residuals


def _get_base_text_features(classnames):

    dataset = "Twitter-COMMs"

    TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device
    )

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:   # text is a domain
            prompt = [template.format(text) for template in TEMPLATES]
            sample = {"text_input": list(prompt)}
            text_features = model.extract_features(sample, mode="text")
            # print(f"sample: {sample}")
            # print(f"text_embeds.shape: {text_features.text_embeds.shape}")
            text_embeds = text_features.text_embeds[:, 0, :]  # [1, 768]
            text_embeddings.append(text_embeds)

    text_embeddings = torch.stack(text_embeddings).mean(1)   # [3, 768]
    return text_embeddings.to(device)