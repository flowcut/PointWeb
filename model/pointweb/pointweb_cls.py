from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointweb.pointweb_module import PointWebSAModule
from model.pointnet2.pointnet2_modules import PointNet2FPModule
from util import pt_util


class PointWebCls(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        c: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, c=3, k=13, use_xyz=True):
        super().__init__()
        self.emb_dims = 512
        self.drop_out = 0.5
        self.output_channels = 40

        self.SA_modules = nn.ModuleList()
        # self.SA_modules.append(PointWebSAModule(npoint=1024, nsample=32, mlp=[c, 32, 32, 64], mlp2=[32, 32], use_xyz=use_xyz))
        # self.SA_modules.append(PointWebSAModule(npoint=256, nsample=32, mlp=[64, 64, 64, 128], mlp2=[32, 32], use_xyz=use_xyz))
        # self.SA_modules.append(PointWebSAModule(npoint=64, nsample=32, mlp=[128, 128, 128, 256], mlp2=[32, 32], use_xyz=use_xyz))
        # self.SA_modules.append(PointWebSAModule(npoint=16, nsample=32, mlp=[256, 256, 256, 512], mlp2=[32, 32], use_xyz=use_xyz))

        self.SA_modules.append(PointWebSAModule(npoint=256, nsample=32, mlp=[c, 32, 32, 64], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=64, nsample=32, mlp=[64, 64, 64, 128], mlp2=[32, 32], use_xyz=use_xyz))
        self.SA_modules.append(PointWebSAModule(npoint=16, nsample=32, mlp=[128, 128, 128, 256], mlp2=[32, 32], use_xyz=use_xyz))


        # self.FP_modules = nn.ModuleList()
        # self.FP_modules.append(PointNet2FPModule(mlp=[128 + c, 128, 128, 128]))
        # self.FP_modules.append(PointNet2FPModule(mlp=[256 + 64, 256, 128]))
        # self.FP_modules.append(PointNet2FPModule(mlp=[256 + 128, 256, 256]))
        # self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, 256]))
        # self.FC_layer = nn.Sequential(pt_util.Conv2d(128, 128, bn=True), nn.Dropout(), pt_util.Conv2d(128, k, activation=None))

        self.conv = nn.Sequential(nn.Conv1d(in_channels=1360, out_channels=self.emb_dims, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(self.emb_dims),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.drop_out)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.drop_out)
        self.linear3 = nn.Linear(256, self.output_channels)
        # self.linear4 = nn.Linear(self.emb_dims, self.output_channels)


    def _break_up_pc(self, pc):
        # print('???:', pc.size())
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        batch_size = pointcloud.size(0)
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        combined = [xyz]
        for i in range(len(self.SA_modules)):
            # print(i,'-l_xyz shape:', l_xyz[i].shape)
            # print(i,'-l_feature shape:', l_features[i].shape)
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            combined.append(li_xyz)
            # combined.append(li_features.max(dim=-1, keepdim=False)[0])
        # print('here!!!', torch.cat((l_xyz + l_features), dim=1).shape)
        combined = torch.cat(combined, dim=1)
        combined = self.conv(combined)
        c1 = F.adaptive_max_pool1d(combined, 1).view(batch_size, -1)
        c2 = F.adaptive_avg_pool1d(combined, 1).view(batch_size, -1)
        combined = torch.cat((c1, c2), 1)

        combined = self.dp1(F.leaky_relu(self.bn1(self.linear1(combined)), negative_slope=0.2))
        combined = self.dp2(F.leaky_relu(self.bn2(self.linear2(combined)), negative_slope=0.2))
        combined = self.linear3(combined)
        # combined = self.linear4(c1)
        return combined 



        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        #return self.FC_layer(l_features[0].unsqueeze(-1)).squeeze(-1)


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
            preds = model(inputs)
            loss = criterion(preds, labels)
            _, classes = torch.max(preds, 1)
            acc = (classes == labels).float().sum() / labels.numel()
            return ModelReturn(preds, loss, {"acc": acc.item(), 'loss': loss.item()})
    return model_fn


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)
            preds = model(inputs)
            loss = criterion(preds, labels)
            _, classes = torch.max(preds, 1)
            acc = (classes == labels).float().sum() / labels.numel()
            return ModelReturn(preds, loss, {"acc": acc.item(), 'loss': loss.item()})
    return model_fn


if __name__ == "__main__":
    import torch.optim as optim
    B, N, C, K = 2, 4096, 3, 13
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.randint(0, 3, (B, N)).cuda()

    model = PointWebSeg(c=C, k=K).cuda()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    model = PointWebSeg(c=C, k=K, use_xyz=False).cuda()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()
