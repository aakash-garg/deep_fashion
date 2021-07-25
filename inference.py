import torch
import os
from PIL import Image
import pandas as pd
from torchvision import transforms

from model import FashionNet
from trainer import ModelUtils
from args import get_args

def generate_output_to_csv(args, model, data_path, transform):
    df = pd.DataFrame(columns=['filename','neck','sleeve_length','pattern'])
    sample_embedding = []
    total_samples = len(os.listdir(data_path))
    for idx, img_name in enumerate(os.listdir(data_path)):
        print("Computing img {}/{}".format(idx+1, total_samples))
        if(img_name[-3:] != 'jpg'):
            continue
        img_path = os.path.join(data_path, img_name)
        img = Image.open(img_path)
        img = transform(img)
        if args.cuda:
            img = img.cuda()
        img = img.unsqueeze(0)

        _, pattern_logits, sleeve_len_logits, neck_type_logits = model(img)
        _, predicted_pattern = torch.max(pattern_logits.data, 1)
        _, predicted_sleeve_len = torch.max(sleeve_len_logits.data, 1)
        _, predicted_neck_type = torch.max(neck_type_logits.data, 1)

        predicted_pattern = predicted_pattern.cpu().detach().numpy()
        predicted_sleeve_len = predicted_sleeve_len.cpu().detach().numpy()
        predicted_neck_type = predicted_neck_type.cpu().detach().numpy()
        row = {
            'filename': img_name,
            'neck': predicted_neck_type[0],
            'sleeve_length': predicted_sleeve_len[0],
            'pattern': predicted_pattern[0]
        }
        df = df.append(row, ignore_index=True)
    df.to_csv('output.csv')

args = get_args()
args.cuda = torch.cuda.is_available() and args.use_gpu
model = FashionNet(args)
model_utils = ModelUtils(args.checkpoint_dir, args, model)
model = model_utils.load_model(best=True)

if args.cuda:
    model = model.cuda()

print("Model loaded...")
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

generate_output_to_csv(args, model, args.infer_dir, transform)
