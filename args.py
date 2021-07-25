from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='FashionClassification')
    ## General arguments of the model
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_pretrained', action="store_true", default=True)
    parser.add_argument('--backbone', type=str, default='resnet50',\
                    choices=['resnet18', 'resnet32' 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])

    ## Dataset args
    parser.add_argument('--data_dir', type=str, default='/home/aakash98/projects/def-chdesa/aakash98/pytorch_code/data')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    ## Arguments Specific to the task heads
    parser.add_argument('--pattern_classes', type=int, default=11) #including the N/A class
    parser.add_argument('--sleeve_len_classes', type=int, default=5)
    parser.add_argument('--neck_type_classes', type=int, default=8)

    ## Training settings
    parser.add_argument('--epochs', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    ## checkpoint Args
    parser.add_argument('--checkpoint_dir', type=str, default='/home/aakash98/projects/def-chdesa/aakash98/pytorch_code/checkpoints')
    parser.add_argument('--resume_training', type=bool, default=False)

    ## To run the inference script (generated csv output from test folder)
    parser.add_argument('--infer_dir', type=str, default='/home/aakash98/projects/def-chdesa/aakash98/pytorch_code/data/sample_data')

    args = parser.parse_args()
    return args
