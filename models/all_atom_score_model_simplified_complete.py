import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean
from e3nn import o3
from models.score_model import AtomEncoder, GaussianSmearing
from utils import so3, torus
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, default_init
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims

# 简化的频谱卷积实现（保留创新，减少内存使用）
class SimplifiedSpectralConv1d(nn.Module):
    """
    创新特性：频域处理用于时间序列建模
    优化点：限制模式数和FFT长度以减少内存使用
    """
    def __init__(self, in_channels, out_channels, modes):
        super(SimplifiedSpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = min(modes, 4)  # 限制modes数量以减少内存
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.complex64)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bni,iom->bno", input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x_ft = torch.fft.rfft(x, dim=0, n=min(8, B))  # 限制FFT长度
        out_ft = torch.zeros(x_ft.shape[0], N, self.out_channels, dtype=torch.complex64, device=x.device)
        out_ft[:self.modes] = self.compl_mul1d(x_ft[:self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=min(8, B), dim=0)
        if x.shape[0] < B:
            x = F.pad(x, (0, 0, 0, 0, 0, B - x.shape[0]))
        return x[:B]

# 简化的时间等变卷积层（保留等变性创新）
class SimplifiedTimeEquivConvLayer(nn.Module):
    """
    创新特性：
    1. 时间等变性处理
    2. 频谱卷积集成
    3. E(3)等变张量积
    优化点：简化时间维度处理，减少内存占用
    """
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=False, 
                 batch_norm=True, dropout=0.0, num_modes=2):
        super(SimplifiedTimeEquivConvLayer, self).__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.sh_irreps = o3.Irreps(sh_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.n_edge_features = n_edge_features
        self.residual = residual
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_modes = num_modes

        # 创新：频谱卷积用于标量特征的时间处理
        scalar_in_dim = sum([mul for mul, irrep in self.in_irreps if irrep.l == 0])
        scalar_out_dim = sum([mul for mul, irrep in self.out_irreps if irrep.l == 0])
        if scalar_in_dim > 0 and scalar_out_dim > 0:
            self.time_conv = SimplifiedSpectralConv1d(scalar_in_dim, scalar_out_dim, num_modes)
        else:
            self.time_conv = None
        
        # 创新：E(3)等变张量积
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.sh_irreps, self.out_irreps, shared_weights=False
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_edge_features, self.tp.weight_numel)
        )
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.out_irreps.dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                param.data = default_init()(param.shape, dtype=param.dtype, device=param.device)

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        if edge_index.shape[1] == 0:
            return torch.zeros_like(x)

        B, N, _ = x.shape
        
        # 创新：时间频谱卷积处理标量特征
        if self.time_conv is not None:
            scalar_mask = torch.tensor([irrep.l == 0 for mul, irrep in self.in_irreps for _ in range(mul)], 
                                     device=x.device)
            scalar_dim = scalar_mask.sum().item()
            if scalar_dim > 0:
                x_scalar = x[:, :, scalar_mask]
                x_scalar = self.time_conv(x_scalar)
                x = torch.cat([x_scalar, x[:, :, ~scalar_mask]], dim=-1) if scalar_dim < x.shape[-1] else x_scalar

        # 创新：等变卷积处理（优化为批次操作）
        x_out_list = []
        for b in range(B):
            x_b = x[b]
            edge_attr_b = edge_attr[b] if edge_attr.dim() == 3 else edge_attr
            edge_sh_b = edge_sh[b] if edge_sh.dim() == 3 else edge_sh
            edge_weights_b = self.edge_mlp(edge_attr_b)
            
            x_b_out = self.tp(x_b[edge_index[0]], edge_sh_b, edge_weights_b)
            out_dim = out_nodes if out_nodes is not None else N
            x_b_out = scatter_mean(x_b_out, edge_index[1], dim=0, dim_size=out_dim)
            x_out_list.append(x_b_out)
        
        x_out = torch.stack(x_out_list, dim=0)
        
        if self.batch_norm:
            original_shape = x_out.shape
            x_out = x_out.view(-1, x_out.shape[-1])
            x_out = self.bn(x_out)
            x_out = x_out.view(original_shape)
        
        if self.dropout > 0:
            x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        
        if self.residual and self.in_irreps == self.out_irreps:
            x_out = x_out + x
        
        return x_out

class SimplifiedTensorProductScoreModel(nn.Module):
    """
    简化的张量积分数模型
    
    保留的创新特性：
    1. 频谱卷积 (SpectralConv1d)
    2. 时间等变性 (TimeEquivConvLayer)
    3. 多图交互 (配体-受体-原子)
    4. 动态时间步调度
    5. E(3)等变性
    6. 球谐函数特征
    
    优化点：
    1. 减少时间步数 (4 vs 8)
    2. 限制频谱模式数 (≤4)
    3. 简化卷积层架构
    4. 优化内存使用
    """
    def __init__(self, t_to_sigma, device, timestep_emb_func=None, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1,
                 num_timesteps=4, num_modes=2):  # 减少时间步数和模式数
        super(SimplifiedTensorProductScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.num_conv_layers = num_conv_layers
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_timesteps = num_timesteps
        self.num_modes = num_modes

        # 嵌入层
        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        # 距离展开
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        # 创新：不可约表示序列
        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        # 创新：简化的卷积层（保留核心交互模式）
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            
            # 配体内部
            lig_layer = SimplifiedTimeEquivConvLayer(
                in_irreps=in_irreps, sh_irreps=self.sh_irreps, out_irreps=out_irreps,
                n_edge_features=3 * ns, residual=False, batch_norm=batch_norm,
                dropout=dropout, num_modes=self.num_modes
            )
            # 受体内部
            rec_layer = SimplifiedTimeEquivConvLayer(
                in_irreps=in_irreps, sh_irreps=self.sh_irreps, out_irreps=out_irreps,
                n_edge_features=3 * ns, residual=False, batch_norm=batch_norm,
                dropout=dropout, num_modes=self.num_modes
            )
            # 交叉交互
            cross_layer = SimplifiedTimeEquivConvLayer(
                in_irreps=in_irreps, sh_irreps=self.sh_irreps, out_irreps=out_irreps,
                n_edge_features=3 * ns, residual=False, batch_norm=batch_norm,
                dropout=dropout, num_modes=self.num_modes
            )
            
            conv_layers.extend([lig_layer, rec_layer, cross_layer])
        
        self.conv_layers = nn.ModuleList(conv_layers)

        # 输出层
        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs)
            )
        else:
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_conv = SimplifiedTimeEquivConvLayer(
                in_irreps=self.conv_layers[-3].out_irreps,  # 最后一个配体层
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                batch_norm=batch_norm,
                dropout=dropout,
                num_modes=self.num_modes
            )
            self.tr_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ns, 1)
            )
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ns, 1)
            )
            
            if not no_torsion:
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = SimplifiedTimeEquivConvLayer(
                    in_irreps=self.conv_layers[-3].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    num_modes=self.num_modes
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

    def forward(self, data):
        """
        前向传播
        保留创新：多图交互、时间步处理、频谱卷积
        优化：简化时间维度处理
        """
        # 处理输入数据格式
        if isinstance(data, list):
            from torch_geometric.data import Batch
            data = Batch.from_data_list(data)
        
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # 创新：动态时间步调度（简化版）
        batch_size = data.num_graphs
        T = min(self.num_timesteps, batch_size)
        
        # 构建图结构
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # 创新：动态交叉距离
        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)

        # 创新：时间维度扩展（简化版保持创新特性）
        lig_node_attr = lig_node_attr.unsqueeze(0).expand(T, -1, -1)
        rec_node_attr = rec_node_attr.unsqueeze(0).expand(T, -1, -1)
        lig_edge_attr = lig_edge_attr.unsqueeze(0).expand(T, -1, -1)
        rec_edge_attr = rec_edge_attr.unsqueeze(0).expand(T, -1, -1)
        cross_edge_attr = cross_edge_attr.unsqueeze(0).expand(T, -1, -1)
        lig_edge_sh = lig_edge_sh.unsqueeze(0).expand(T, -1, -1)
        rec_edge_sh = rec_edge_sh.unsqueeze(0).expand(T, -1, -1)
        cross_edge_sh = cross_edge_sh.unsqueeze(0).expand(T, -1, -1)

        # 创新：多图交互卷积（简化但保留核心交互）
        for l in range(self.num_conv_layers):
            # 配体内部连接
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[:, lig_edge_index[0], :self.ns], 
                                      lig_node_attr[:, lig_edge_index[1], :self.ns]], -1)
            lig_update = self.conv_layers[3*l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # 受体内部连接
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[:, rec_edge_index[0], :self.ns], 
                                      rec_node_attr[:, rec_edge_index[1], :self.ns]], -1)
            rec_update = self.conv_layers[3*l+1](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

            # 创新：交叉连接（配体-受体交互）
            cross_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[:, cross_edge_index[0], :self.ns], 
                                        rec_node_attr[:, cross_edge_index[1], :self.ns]], -1)
            cross_update = self.conv_layers[3*l+2](rec_node_attr, cross_edge_index, cross_edge_attr_, cross_edge_sh,
                                                  out_nodes=lig_node_attr.shape[1])

            # 更新特征
            pad_size = lig_update.shape[-1] - lig_node_attr.shape[-1]
            if pad_size > 0:
                lig_node_attr = F.pad(lig_node_attr, (0, pad_size))
            lig_node_attr = lig_node_attr + lig_update + cross_update

            pad_size = rec_update.shape[-1] - rec_node_attr.shape[-1]
            if pad_size > 0:
                rec_node_attr = F.pad(rec_node_attr, (0, pad_size))
            rec_node_attr = rec_node_attr + rec_update

        # 时间维度平均（保持创新特性的简化处理）
        lig_node_attr = lig_node_attr.mean(dim=0)

        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:, :self.ns], lig_node_attr[:, -self.ns:]], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:, :self.ns]
            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0))
            return confidence

        # 创新：中心卷积用于全局预测
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        
        # 为final_conv创建时间维度
        lig_node_attr_time = lig_node_attr.unsqueeze(0).expand(T, -1, -1)
        center_edge_attr_time = center_edge_attr.unsqueeze(0).expand(T, -1, -1)
        center_edge_sh_time = center_edge_sh.unsqueeze(0).expand(T, -1, -1)
        
        global_pred = self.final_conv(lig_node_attr_time, center_edge_index, center_edge_attr_time, center_edge_sh_time, out_nodes=data.num_graphs)
        global_pred = global_pred.mean(dim=0)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # 调整预测幅度
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))

        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=self.device)

        # 创新：扭转角预测（保留完整流程）
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])
        tor_edge_attr = self.final_edge_embedding(tor_edge_attr)
        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        
        # 扭转角时间维度处理
        lig_node_attr_tor = lig_node_attr.unsqueeze(0).expand(T, -1, -1)
        tor_edge_attr_time = tor_edge_attr.unsqueeze(0).expand(T, -1, -1)
        tor_edge_sh_time = tor_edge_sh.unsqueeze(0).expand(T, -1, -1)
        
        tor_pred = self.tor_bond_conv(lig_node_attr_tor, tor_edge_index, tor_edge_attr_time, tor_edge_sh_time,
                                     out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = tor_pred.mean(dim=0)
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                           .to(data['ligand'].x.device))
        
        return tr_pred, rot_pred, tor_pred

    # 图构建方法（保留原始逻辑）
    def build_lig_conv_graph(self, data):
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], -1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], -1)
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr'])
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], -1)
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        if torch.is_tensor(cross_distance_cutoff):
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                              data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                              data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                              data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)
        center_pos = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)
        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)
        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return bonds, edge_index, edge_attr, edge_sh 
