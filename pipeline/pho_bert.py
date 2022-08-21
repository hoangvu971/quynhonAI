import torch
from transformers import AutoModel, AutoTokenizer

from segmentation import VnCoreNLPHelper


if __name__ == "__main__":
    core_nlp = VnCoreNLPHelper()
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
    segmented = core_nlp.model.word_segment(text)

    input_ids = torch.tensor([tokenizer.encode(segmented)])
    with torch.no_grad():
        features = phobert(input_ids)
        print(features)