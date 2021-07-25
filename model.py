import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler

# from create_dataset import *
from utils import *
from args import get_args

# Multi Task Learning Architecture for Garment Classification
class FashionNet(nn.Module):
    def __init__(self, args):
        super(FashionNet, self).__init__()
        resnet_base_model = create_backbone(args.backbone, 1000, args.use_pretrained) # 1000 is dummy num_classes (will be discarded later)
        # self._print_resnet_children(resnet_base_model)
        self.base_model = self._discard_last_basicblock_from_resnet(resnet_base_model)
        # self._print_resnet_children(self.base_model)

        # Defining task heads (fe denotes 'features_extracted')
        if args.backbone in ['resnet18', 'resnet32']:
            task_head_channels = 512
        elif args.backbone in ['resnet50', 'resnet101', 'resnet152']:
            task_head_channels = 2048
        self.pattern_fe = self._get_task_head_arch(in_channels=task_head_channels, out_channels=task_head_channels)
        self.sleeve_len_fe = self._get_task_head_arch(in_channels=task_head_channels, out_channels=task_head_channels)
        self.neck_type_fe = self._get_task_head_arch(in_channels=task_head_channels, out_channels=task_head_channels)

        self.pattern_classifier = nn.Linear(task_head_channels, args.pattern_classes)
        self.sleeve_len_classifier = nn.Linear(task_head_channels, args.sleeve_len_classes)
        self.neck_type_classifier = nn.Linear(task_head_channels, args.neck_type_classes)

    def _get_task_head_arch(self, in_channels, out_channels):
        task_head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.AdaptiveAvgPool2d((1,1)),
            # nn.Linear(out_channels, task_num_classes)
        )
        return task_head

    def _print_resnet_children(self, base_model):
        child_counter = 0
        for child in base_model.children():
            print(" Child", child_counter, "is: ")
            print(child)
            child_counter +=1

    def _discard_last_basicblock_from_resnet(self, base_model):
        blocks_to_keep = []
        child_counter = 0
        for child in base_model.children():
            ## child_counter 8 denotes the adaptive avg pooling layer
            if(child_counter == 8):
                break
            else:
                sub_children = list(child.children())
                if(len(sub_children) == 0): # for the initial layers which don't use BasicBlock
                    blocks_to_keep.append(child)
                else:
                    for children_of_child in sub_children:
                        blocks_to_keep.append(children_of_child)
            child_counter += 1

        truncated_model = nn.Sequential(*blocks_to_keep[:-1])
        return truncated_model

    def forward(self, input):
        main_output = self.base_model(input)
        pattern_fe = torch.flatten(self.pattern_fe(main_output), 1)
        sleeve_len_fe = torch.flatten(self.sleeve_len_fe(main_output), 1)
        neck_type_fe = torch.flatten(self.neck_type_fe(main_output), 1)

        pattern = self.pattern_classifier(pattern_fe)
        sleeve_len = self.sleeve_len_classifier(sleeve_len_fe)
        neck_type = self.neck_type_classifier(neck_type_fe)

        return main_output, pattern, sleeve_len, neck_type

if __name__ == "__main__":
    args = get_args()
    net = FashionNet(args).cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    out, pattern, sleeve_len, neck_type = net(x)

    print(out.shape) # from base -> ([batch_size, 512, 7, 7]) (global feature maps)
    print(pattern.shape) # shape: [batch_size, pattern_classes]
    print(sleeve_len.shape)
    print(neck_type.shape)
