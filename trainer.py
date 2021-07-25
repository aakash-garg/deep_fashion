import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FashionDataset

class ModelUtils():
    def __init__(self, checkpoint_dir, args, model):
        self.network = model
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(args.seed))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save_model(self, model, best=False):
        if best:
            print('Saved best model..... ')
            torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'best.pth'))
        else:
            print('Saving current model to "{}"'.format(self.checkpoint_dir))
            torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'current.pth'))

    def load_model(self, best=False):
        if best:
            print("Loading testing model.... ")
            self.network.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, 'best.pth')))
        else:
            print("Resuming checkpoint.... ")
            self.network.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, 'current.pth')))
        return self.network

class Trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.model_utils = ModelUtils(args.checkpoint_dir, args, model)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_set = FashionDataset(args, train_transform, phase='train')
        val_set = FashionDataset(args, test_transform, phase='val')
        test_set = FashionDataset(args, test_transform, phase='test')

        self.train_loader = DataLoader(dataset=train_set,\
                batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(dataset=val_set,\
                batch_size=args.batch_size, num_workers=0)
        self.test_loader = DataLoader(dataset=val_set,\
                batch_size=args.batch_size, num_workers=0)

        self.criterion = nn.CrossEntropyLoss()

        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        loss_fnc = nn.CrossEntropyLoss()
        if self.args.resume_training:
            try:
                self.model = self.model_utils.load_model(best=False)
            except:
                print("Resuming checkpoint failed...")
                pass

        best_val_acc = 0

        for epoch in range(self.args.epochs):
            print("################################")
            print("Running training epoch: {}/{}".format(epoch, self.args.epochs))
            running_pattern_loss = 0
            running_sleeve_len_loss = 0
            running_neck_type_loss = 0
            self.model.train()

            for index, (images, pattern_label, sleeve_len_label, neck_type_label) in enumerate(self.train_loader):
                print("Batch {}/{}".format(index, len(self.train_loader)))
                if self.args.cuda:
                    images, pattern_label, sleeve_len_label, neck_type_label =\
                                images.cuda(), pattern_label.cuda(), sleeve_len_label.cuda(), neck_type_label.cuda()

                _, pattern_logits, sleeve_len_logits, neck_type_logits = self.model(images)

                optimizer.zero_grad()
                ## Accumulating the gradients for each task
                pattern_loss = self.criterion(pattern_logits, pattern_label)
                pattern_loss.backward(retain_graph=True)
                running_pattern_loss += pattern_loss.item()

                sleeve_len_loss = self.criterion(sleeve_len_logits, sleeve_len_label)
                sleeve_len_loss.backward(retain_graph=True)
                running_sleeve_len_loss += sleeve_len_loss.item()

                neck_type_loss = self.criterion(neck_type_logits, neck_type_label)
                neck_type_loss.backward()
                running_neck_type_loss += neck_type_loss.item()

                optimizer.step()

            scheduler.step()
            
            print('Training Loss (Pattern): {:.2f}'.format(
                    running_pattern_loss / len(self.train_loader.dataset),
            ))
            print('Training Loss (Sleeve loss): {:.2f}'.format(
                    running_sleeve_len_loss / len(self.train_loader.dataset),
            ))
            print('Training Loss (Pattern): {:.2f}'.format(
                    running_neck_type_loss / len(self.train_loader.dataset),
            ))

            acc_pattern, acc_sleeve_len, acc_neck_type = self.evaluate(self.val_loader, phase='val')
            avg_acc = (acc_pattern + acc_sleeve_len + acc_neck_type)/3.0
            if(avg_acc > best_val_acc):
                self.model_utils.save_model(self.model, best=True)
                best_val_acc = avg_acc
            else:
                self.model_utils.save_model(self.model, best=False)

    def evaluate(self, dataloader, phase='val', load_model=False):
        if phase=='test' and load_model:
            self.model_utils.load_model(best=True)

        self.model.eval()
        correct_pred_pattern = 0
        correct_pred_sleeve_len = 0
        correct_pred_neck_type = 0

        with torch.no_grad():
            if phase == 'val':
                print("Evaluating on validation set....")
            if phase == 'test':
                print("Evaluating on test set....")

            for index, (images, pattern_label, sleeve_len_label, neck_type_label) in enumerate(dataloader):
                print("Batch {}/{}".format(index, len(dataloader)))
                if self.args.cuda:
                    images, pattern_label, sleeve_len_label, neck_type_label =\
                                images.cuda(), pattern_label.cuda(), sleeve_len_label.cuda(), neck_type_label.cuda()

                _, pattern_logits, sleeve_len_logits, neck_type_logits = self.model(images)
                _, predicted_pattern = torch.max(pattern_logits.data, 1)
                _, predicted_sleeve_len = torch.max(sleeve_len_logits.data, 1)
                _, predicted_neck_type = torch.max(neck_type_logits.data, 1)

                correct_pred_pattern += torch.sum(predicted_pattern == pattern_label)
                correct_pred_sleeve_len += torch.sum(predicted_sleeve_len == sleeve_len_label)
                correct_pred_neck_type += torch.sum(predicted_neck_type == neck_type_label)

            acc_pattern = 100. * correct_pred_pattern / len(dataloader.dataset)
            acc_sleeve_len = 100. * correct_pred_sleeve_len / len(dataloader.dataset)
            acc_neck_type = 100. * correct_pred_neck_type / len(dataloader.dataset)

            print('Pattern Accuracy: {}/{} ({:.2f}%)'.format(
                    correct_pred_pattern,
                    len(dataloader.dataset),
                    100. * correct_pred_pattern / len(dataloader.dataset)
            ))
            print('Sleeve length Accuracy: {}/{} ({:.2f}%)'.format(
                    correct_pred_sleeve_len,
                    len(dataloader.dataset),
                    100. * correct_pred_sleeve_len / len(dataloader.dataset)
            ))
            print('Neck type Accuracy: {}/{} ({:.2f}%)'.format(
                    correct_pred_neck_type,
                    len(dataloader.dataset),
                    100. * correct_pred_neck_type / len(dataloader.dataset)
            ))

            return acc_pattern, acc_sleeve_len, acc_neck_type
