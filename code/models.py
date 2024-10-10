import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from FocusedLinearAttention import SiameseFLA
from coordconv import CoordConv1d

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class MIFDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(MIFDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_layers = config["DRUG"]["LAYERS"]
        drug_num_head = config["DRUG"]["NUM_HEAD"]
        drug_padding = config["DRUG"]["PADDING"]

        protein_layers = config["PROTEIN"]["LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        protein_padding = config["PROTEIN"]["PADDING"]

        siamese_fla_emb_dim = config["SIAMESEFLA"]["EMBEDDING_DIM"]


        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.smiles_extractor = MolecularCoordconv(dim=drug_embedding, padding=drug_padding)
        self.protein_extractor = ProteinCoordconv(dim=protein_emb_dim, padding=protein_padding)
        #Cross-FLA
        self.siamesefla = SiameseFLA(siamese_fla_emb_dim)
        #MLPDecoder
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)


    def forward(self, smi_d, bg_d, v_p, mode="train"):
        #Drug Encoder
        v_d = self.drug_extractor(bg_d)
        v_s = self.smiles_extractor(smi_d)
        #Protein Enccoder
        v_p = self.protein_extractor(v_p)
        #Feature Fusion
        f = self.siamesefla(v_d, v_s, v_p)
        att = None
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_s, v_p, f, score
        elif mode == "eval":
            return v_d, v_s, v_p, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class ProteinCoordconv(nn.Module):
    def __init__(self, dim, kernels=[3, 6, 9], padding=True):
        super(ProteinCoordconv, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, dim)
        self.conv1 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        return x

class MolecularCoordconv(nn.Module):
    def __init__(self, dim, kernels=[3, 6, 9], padding=True):
        super(MolecularCoordconv, self).__init__()
        if padding:
            self.embedding = nn.Embedding(65, dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(65, dim)
        self.conv1 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        return x
#
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x




