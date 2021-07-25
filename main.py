import torch
from args import get_args
from model import FashionNet
from trainer import Trainer

if __name__ == "__main__":
    args = get_args()
    args.cuda = torch.cuda.is_available() and args.use_gpu
    print("Using experiment seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    network = FashionNet(args)
    trainer = Trainer(args, network)
    if (args.phase == 'train'):
        trainer.train()
    elif (args.phase == 'test'):
        trainer.evaluate(trainer.test_loader, phase='test', load_model=True)
