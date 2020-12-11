import torch
import torch.nn as nn
import torchvision.models as models
import pretrainedmodels

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


class pret_torch_nets(nn.Module):
    def __init__(self, model_name, pretrained=False, class_num=1000):
        super(pret_torch_nets, self).__init__()

        # example
        # assert inceptionv4(num_classes=10, pretrained=None)
        # assert inceptionv4(num_classes=1000, pretrained='imagenet')
        # assert inceptionv4(num_classes=1001, pretrained='imagenet+background')

        if pretrained:
            self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            print("Creating model '%s' with pretrained weights from pretrainedmodels" % model_name)
        else:
            self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            print("Creating model '%s' from scratch" % model_name)

        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, class_num)
        self.model.last_linear.apply(init_weights)

    def forward(self, x):
        return self.model(x)

    def get_parameters(self):
        raise AssertionError('get_parameters is not supported')

    # def get_input_size(self):
    #     # [3, 299, 299] for inception* networks
    #     # [3, 224, 224] for resnet* networks
    #     return self.model.input_size()
    #
    # def get_input_space(self):
    #     # str: RGB or BGR
    #     return self.model.input_space()
    #
    # def get_input_range(self):
    #     # [0, 1] or [0, 255]
    #     return self.model.input_range()
    #
    # def get_mean(self):
    #     # [0.5, 0.5, 0.5] for inception * networks,
    #     # [0.485, 0.456, 0.406]     for resnet * networks.
    #     return self.model.mean()
    #
    # def get_std(self):
    #     # [0.5, 0.5, 0.5] for inception * networks,
    #     # [0.229, 0.224, 0.225] for resnet * networks.
    #     return self.model.std()


class resnet(nn.Module):
    def __init__(self, resnet_name, pretrainedCaffe=False, pretrainedTorch=False, class_num=1000):
        super(resnet, self).__init__()

        if pretrainedTorch and pretrainedCaffe:
            raise AssertionError('pretrainedCaffe & pretrainedTorch cannot be True at the sametime!')
        elif pretrainedTorch:
            model = models.__dict__[resnet_name](pretrained=True)
            print("Creating model '%s' with pretrained weights from torch" % resnet_name)
        elif pretrainedCaffe:
            model = models.__dict__[resnet_name]()

            if resnet_name == "resnet18":
                path_pretrainedCaffe = 'pretrained/resnet18_caffe.pth'
            elif resnet_name == "resnet50":
                path_pretrainedCaffe = 'pretrained/resnet50_caffe.pth'
            elif resnet_name == "resnet101":
                path_pretrainedCaffe = 'pretrained/resnet101_caffe.pth'
            else:
                raise AssertionError('%s is not supported' % resnet_name)

            print("Creating model '%s' with pretrained weights from %s" % (resnet_name, path_pretrainedCaffe))
            old_state_dict = torch.load(path_pretrainedCaffe)
            new_state_dict = model.state_dict()  # all things are copied, key and value of target net

            # change pretrained-model-name to match with the FRCN
            for key, value in old_state_dict.items():  # ex: encoder.conv1. weight, .. projector.0.weight
                new_key = key  # this could be 'conv1' or 'projector'

                if new_key in new_state_dict:
                    new_state_dict[new_key] = value
                else:
                    print('\t[%s] key is ignored because encoder is not included' % key)

            model.load_state_dict({k: v for k, v in new_state_dict.items() if k in model.state_dict()})
        else:
            print("Creating model '%s' from scratch" % resnet_name)
            model = models.__dict__[resnet_name]()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.fc = nn.Linear(model.fc.in_features, class_num)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_parameters(self):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]

        return parameter_list





class resnet_multi(nn.Module):
    def __init__(self, resnet_name, pretrainedCaffe=False, pretrainedTorch=False, class_num1=1000, class_num2=1000, class_num3=1000):
        super(resnet_multi, self).__init__()

        if pretrainedTorch and pretrainedCaffe:
            raise AssertionError('pretrainedCaffe & pretrainedTorch cannot be True at the sametime!')
        elif pretrainedTorch:
            model = models.__dict__[resnet_name](pretrained=True)
            print("Creating model '%s' with pretrained weights from torch" % resnet_name)
        elif pretrainedCaffe:
            model = models.__dict__[resnet_name]()

            if resnet_name == "resnet18":
                path_pretrainedCaffe = 'pretrained/resnet18_caffe.pth'
            elif resnet_name == "resnet50":
                path_pretrainedCaffe = 'pretrained/resnet50_caffe.pth'
            elif resnet_name == "resnet101":
                path_pretrainedCaffe = 'pretrained/resnet101_caffe.pth'
            else:
                raise AssertionError('%s is not supported' % resnet_name)

            print("Creating model '%s' with pretrained weights from %s" % (resnet_name, path_pretrainedCaffe))
            old_state_dict = torch.load(path_pretrainedCaffe)
            new_state_dict = model.state_dict()  # all things are copied, key and value of target net

            # change pretrained-model-name to match with the FRCN
            for key, value in old_state_dict.items():  # ex: encoder.conv1. weight, .. projector.0.weight
                new_key = key  # this could be 'conv1' or 'projector'

                if new_key in new_state_dict:
                    new_state_dict[new_key] = value
                else:
                    print('\t[%s] key is ignored because encoder is not included' % key)

            model.load_state_dict({k: v for k, v in new_state_dict.items() if k in model.state_dict()})
        else:
            print("Creating model '%s' from scratch" % resnet_name)
            model = models.__dict__[resnet_name]()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.fc1 = nn.Linear(model.fc.in_features, class_num1)
        self.fc1.apply(init_weights)

        self.fc2 = nn.Linear(model.fc.in_features, class_num2)
        self.fc2.apply(init_weights)

        self.fc3 = nn.Linear(model.fc.in_features, class_num3)
        self.fc3.apply(init_weights)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)

        return x1, x2, x3

    def get_parameters(self):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.fc1.parameters(), "lr_mult": 10, 'decay_mult': 2}, \
                          {"params": self.fc2.parameters(), "lr_mult": 10, 'decay_mult': 2}, \
                          {"params": self.fc3.parameters(), "lr_mult": 10, 'decay_mult': 2}
                          ]

        return parameter_list


class pret_torch_nets_multi(nn.Module):
    def __init__(self, model_name, pretrained=False, class_num1=1000, class_num2=1000, class_num3=1000):
        super(pret_torch_nets_multi, self).__init__()

        # example
        # assert inceptionv4(num_classes=10, pretrained=None)
        # assert inceptionv4(num_classes=1000, pretrained='imagenet')
        # assert inceptionv4(num_classes=1001, pretrained='imagenet+background')

        if pretrained:
            self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            print("Creating model '%s' with pretrained weights from pretrainedmodels" % model_name)
        else:
            self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            print("Creating model '%s' from scratch" % model_name)

        num_ftrs = self.model.last_linear.in_features
        self.model.last_linear = nn.Identity()

        self.fc1 = nn.Linear(num_ftrs, class_num1)
        self.fc1.apply(init_weights)

        self.fc2 = nn.Linear(num_ftrs, class_num2)
        self.fc2.apply(init_weights)

        self.fc3 = nn.Linear(num_ftrs, class_num3)
        self.fc3.apply(init_weights)

    def forward(self, x):
        x = self.model(x)

        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)

        return x1, x2, x3

    def get_parameters(self):
        raise AssertionError('get_parameters is not supported')
