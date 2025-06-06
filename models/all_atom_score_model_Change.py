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

# 频谱卷积实现
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.complex64)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("tni,iom->tno", input, weights)

    def forward(self, x):
        T, N, C = x.shape
        x_ft = torch.fft.rfft(x, dim=0)
        out_ft = torch.zeros(T // 2 + 1, N, self.out_channels, dtype=torch.complex64, device=x.device)
        out_ft[:self.modes] = self.compl_mul1d(x_ft[:self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=T, dim=0)
        return x

# 完善的 TimeEquivConvLayer
class TimeEquivConvLayer(nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=False, 
                 batch_norm=True, dropout=0.0, num_timesteps=8, num_modes=2):
        super(TimeEquivConvLayer, self).__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.sh_irreps = o3.Irreps(sh_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.n_edge_features = n_edge_features
        self.residual = residual
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.num_modes = num_modes

        # 标量特征的频谱卷积
        scalar_in_dim = sum([mul for mul, irrep in self.in_irreps if irrep.l == 0])
        scalar_out_dim = sum([mul for mul, irrep in self.out_irreps if irrep.l == 0])
        self.time_conv = SpectralConv1d(scalar_in_dim, scalar_out_dim, num_modes) if scalar_in_dim > 0 else None
        
        # 等变张量积
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.sh_irreps, self.out_irreps, shared_weights=False
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_edge_features, self.tp.weight_numel)
        )
        
        # 批归一化
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.out_irreps.dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                param.data = default_init()(param.shape, dtype=param.dtype, device=param.device)

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        if edge_index.shape[1] == 0:
            return torch.zeros_like(x)  # 空图处理

        T, N, _ = x.shape
        assert edge_attr.shape[0] == T and edge_sh.shape[0] == T, "Time dimension mismatch"
        
        # 分离标量和向量特征
        scalar_mask = torch.tensor([irrep.l == 0 for mul, irrep in self.in_irreps for _ in range(mul)], 
                                 device=x.device)
        scalar_dim = scalar_mask.sum().item()
        x_scalar = x[:, :, scalar_mask] if scalar_dim > 0 else None
        x_vector = x[:, :, ~scalar_mask] if scalar_dim < x.shape[-1] else None

        # 标量特征通过频谱卷积
        if x_scalar is not None and self.time_conv is not None:
            x_scalar = self.time_conv(x_scalar)

        # 边特征处理
        edge_weights = self.edge_mlp(edge_attr)
        
        # 等变卷积
        x_out = []
        for t in range(T):
            x_t = x[t]
            edge_sh_t = edge_sh[t]
            edge_weights_t = edge_weights[t]
            x_t_out = self.tp(x_t[edge_index[0]], edge_sh_t, edge_weights_t)
            
            out_dim = out_nodes if out_nodes is not None else N
            x_t_out = scatter_mean(x_t_out, edge_index[1], dim=0, dim_size=out_dim)
            x_out.append(x_t_out)
        
        x_out = torch.stack(x_out, dim=0)
        
        # 融合标量和向量输出
        scalar_out_dim = sum([mul for mul, irrep in self.out_irreps if irrep.l == 0])
        if x_scalar is not None:
            x_out_scalar = x_out[:, :, :scalar_out_dim] + x_scalar
            x_out = torch.cat([x_out_scalar, x_out[:, :, scalar_out_dim:]], dim=-1)
        
        # 批归一化和 dropout
        if self.batch_norm:
            x_out = x_out.permute(1, 2, 0)
            x_out = self.bn(x_out.reshape(-1, x_out.shape[1])).reshape(x_out.shape)
            x_out = x_out.permute(2, 0, 1)
        
        if self.dropout > 0:
            x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        
        # 残差连接
        if self.residual and self.in_irreps == self.out_irreps:
            x_out = x_out + x
        
        return x_out

class TensorProductScoreModel(nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func=None, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1,
                 num_timesteps=8, num_modes=2, schedule_type='cosine'):
        super(TensorProductScoreModel, self).__init__()
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
        self.timestep_emb_func = timestep_emb_func  # 外部传入的时间嵌入函数
        self.confidence_mode = confidence_mode
        self.num_timesteps = num_timesteps
        self.num_modes = num_modes
        self.schedule_type = schedule_type

        # 动态调整时间步数
        self.adjust_timesteps()

        # embedding layers
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
        self.atom_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.atom_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.lr_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.ar_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.la_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

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

        # convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout,
                'num_timesteps': self.num_timesteps,
                'num_modes': self.num_modes
            }
            for _ in range(9):
                conv_layers.append(TimeEquivConvLayer(**parameters))
        self.conv_layers = nn.ModuleList(conv_layers)

        # confidence and affinity prediction
        if self.confidence_mode:
            output_confidence_dim = num_confidence_outputs
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, 2 * output_confidence_dim)  # Output mean and variance for CFR
            )
        else:
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_conv = TimeEquivConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                batch_norm=batch_norm,
                dropout=dropout,
                num_timesteps=self.num_timesteps,
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
                self.tor_bond_conv = TimeEquivConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    num_timesteps=self.num_timesteps,
                    num_modes=self.num_modes
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

    def adjust_timesteps(self, num_nodes=None):
        """根据分子复杂性动态调整时间步数"""
        if num_nodes is not None:
            # 根据节点数动态调整，假设大型分子需要更多时间步
            self.num_timesteps = max(4, min(16, int(math.log2(num_nodes) * 2)))
        else:
            self.num_timesteps = max(4, min(16, self.num_conv_layers * 4))

    def expand_edge_index(self, edge_index, T):
        """优化边索引扩展，减少内存占用"""
        if edge_index.shape[1] == 0:
            return edge_index
        # 使用稀疏表示，避免大张量
        edge_index_expanded = edge_index.unsqueeze(1).expand(-1, T, -1).clone()
        edge_index_expanded = edge_index_expanded.view(2, -1)
        return edge_index_expanded

    def forward(self, data):
        # 处理DataListLoader返回的图对象列表和常规DataLoader返回的批处理图
        if isinstance(data, list):
            # DataListLoader返回的是图对象列表，需要合并为批处理
            from torch_geometric.data import Batch
            data = Batch.from_data_list(data)
        
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # 动态调整时间步数（基于配体节点数）
        self.adjust_timesteps(num_nodes=data['ligand'].pos.shape[0])

        # 时间步调度
        T = self.num_timesteps
        timesteps = torch.tensor(get_t_schedule(T, self.schedule_type), device=self.device)
        time_emb = self.timestep_emb_func(timesteps, self.sigma_embed_dim)
        assert time_emb.shape == (T, self.sigma_embed_dim), f"Expected time_emb shape [{T}, {self.sigma_embed_dim}], got {time_emb.shape}"
        time_emb = time_emb.unsqueeze(1)

        # build graphs
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        lig_edge_index = self.expand_edge_index(lig_edge_index, T)
        lig_edge_attr = lig_edge_attr.unsqueeze(0).expand(T, -1, -1)
        lig_edge_sh = lig_edge_sh.unsqueeze(0).expand(T, -1, -1)

        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        rec_edge_index = self.expand_edge_index(rec_edge_index, T)
        rec_edge_attr = rec_edge_attr.unsqueeze(0).expand(T, -1, -1)
        rec_edge_sh = rec_edge_sh.unsqueeze(0).expand(T, -1, -1)

        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(data)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)
        atom_edge_index = self.expand_edge_index(atom_edge_index, T)
        atom_edge_attr = atom_edge_attr.unsqueeze(0).expand(T, -1, -1)
        atom_edge_sh = atom_edge_sh.unsqueeze(0).expand(T, -1, -1)

        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        lr_edge_index, lr_edge_attr, lr_edge_sh, la_edge_index, la_edge_attr, \
            la_edge_sh, ar_edge_index, ar_edge_attr, ar_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        lr_edge_attr = self.lr_edge_embedding(lr_edge_attr)
        lr_edge_index = self.expand_edge_index(lr_edge_index, T)
        lr_edge_attr = lr_edge_attr.unsqueeze(0).expand(T, -1, -1)
        lr_edge_sh = lr_edge_sh.unsqueeze(0).expand(T, -1, -1)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)
        la_edge_index = self.expand_edge_index(la_edge_index, T)
        la_edge_attr = la_edge_attr.unsqueeze(0).expand(T, -1, -1)
        la_edge_sh = la_edge_sh.unsqueeze(0).expand(T, -1, -1)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)
        ar_edge_index = self.expand_edge_index(ar_edge_index, T)
        ar_edge_attr = ar_edge_attr.unsqueeze(0).expand(T, -1, -1)
        ar_edge_sh = ar_edge_sh.unsqueeze(0).expand(T, -1, -1)

        # 时间步权重
        time_weights = 1 - timesteps  # [T]，早期时间步权重较低
        time_weights = time_weights.view(T, 1, 1)  # [T, 1, 1]

        # 卷积层
        for l in range(self.num_conv_layers):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[:, lig_edge_index[0], :self.ns], 
                                      lig_node_attr[:, lig_edge_index[1], :self.ns]], -1)
            lig_update = self.conv_layers[9*l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            lr_edge_attr_ = torch.cat([lr_edge_attr, lig_node_attr[:, lr_edge_index[0], :self.ns], 
                                     rec_node_attr[:, lr_edge_index[1], :self.ns]], -1)
            lr_update = self.conv_layers[9*l+1](rec_node_attr, lr_edge_index, lr_edge_attr_, lr_edge_sh,
                                               out_nodes=lig_node_attr.shape[1])

            la_edge_attr_ = torch.cat([la_edge_attr, lig_node_attr[:, la_edge_index[0], :self.ns], 
                                     atom_node_attr[:, la_edge_index[1], :self.ns]], -1)
            la_update = self.conv_layers[9*l+2](atom_node_attr, la_edge_index, la_edge_attr_, la_edge_sh,
                                               out_nodes=lig_node_attr.shape[1])

            # Always execute all conv layers to ensure parameter usage
            atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[:, atom_edge_index[0], :self.ns], 
                                      atom_node_attr[:, atom_edge_index[1], :self.ns]], -1)
            atom_update = self.conv_layers[9*l+3](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh)

            al_edge_attr_ = torch.cat([la_edge_attr, atom_node_attr[:, la_edge_index[1], :self.ns], 
                                     lig_node_attr[:, la_edge_index[0], :self.ns]], -1)
            al_update = self.conv_layers[9*l+4](lig_node_attr, torch.flip(la_edge_index, dims=[0]), 
                                               al_edge_attr_, la_edge_sh, out_nodes=atom_node_attr.shape[1])

            ar_edge_attr_ = torch.cat([ar_edge_attr, atom_node_attr[:, ar_edge_index[0], :self.ns], 
                                     rec_node_attr[:, ar_edge_index[1], :self.ns]], -1)
            ar_update = self.conv_layers[9*l+5](rec_node_attr, ar_edge_index, ar_edge_attr_, ar_edge_sh,
                                               out_nodes=atom_node_attr.shape[1])

            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[:, rec_edge_index[0], :self.ns], 
                                      rec_node_attr[:, rec_edge_index[1], :self.ns]], -1)
            rec_update = self.conv_layers[9*l+6](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

            rl_edge_attr_ = torch.cat([lr_edge_attr, rec_node_attr[:, lr_edge_index[1], :self.ns], 
                                     lig_node_attr[:, lr_edge_index[0], :self.ns]], -1)
            rl_update = self.conv_layers[9*l+7](lig_node_attr, torch.flip(lr_edge_index, dims=[0]), 
                                               rl_edge_attr_, lr_edge_sh, out_nodes=rec_node_attr.shape[1])

            ra_edge_attr_ = torch.cat([ar_edge_attr, rec_node_attr[:, ar_edge_index[1], :self.ns], 
                                     atom_node_attr[:, ar_edge_index[0], :self.ns]], -1)
            ra_update = self.conv_layers[9*l+8](atom_node_attr, torch.flip(ar_edge_index, dims=[0]), 
                                               ra_edge_attr_, ar_edge_sh, out_nodes=rec_node_attr.shape[1])

            # Conditional updates based on layer (only apply when not last layer for some updates)
            if l != self.num_conv_layers - 1:
                # 更新节点特征（加权）
                pad_size = atom_update.shape[-1] - atom_node_attr.shape[-1]
                if pad_size > 0:
                    atom_node_attr = F.pad(atom_node_attr, (0, pad_size))
                atom_node_attr = atom_node_attr + time_weights * (atom_update + al_update + ar_update)

                pad_size = rec_update.shape[-1] - rec_node_attr.shape[-1]
                if pad_size > 0:
                    rec_node_attr = F.pad(rec_node_attr, (0, pad_size))
                rec_node_attr = rec_node_attr + time_weights * (rec_update + ra_update + rl_update)

            pad_size = lig_update.shape[-1] - lig_node_attr.shape[-1]
            if pad_size > 0:
                lig_node_attr = F.pad(lig_node_attr, (0, pad_size))
            lig_node_attr = lig_node_attr + time_weights * (lig_update + la_update + lr_update)

        # confidence and affinity prediction
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:, :, :self.ns], lig_node_attr[:, :, -self.ns:]], dim=-1) if self.num_conv_layers >= 3 else lig_node_attr[:, :, :self.ns]
            scalar_lig_attr = (scalar_lig_attr * time_weights).mean(dim=0)
            confidence_output = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0))
            # Split output into mean and variance for CFR-like uncertainty quantification
            output_confidence_dim = confidence_output.shape[-1] // 2  # Half for mean, half for variance
            mean = confidence_output[:, :output_confidence_dim]
            variance = F.softplus(confidence_output[:, output_confidence_dim:])  # Ensure variance is positive
            confidence = (mean, variance)  # Return tuple of mean and variance
            return confidence

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_index = self.expand_edge_index(center_edge_index, T)
        center_edge_attr = center_edge_attr.unsqueeze(0).expand(T, -1, -1)
        center_edge_sh = center_edge_sh.unsqueeze(0).expand(T, -1, -1)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[:, center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
        global_pred = (global_pred * time_weights).mean(dim=0)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(torch.tensor([0]).to(self.device), self.sigma_embed_dim)

        # adjust the magnitude of the score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))

        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0:
            # 即使没有扭转角，也要确保所有参数都参与计算
            # 创建一个虚拟的扭转角预测，这样所有卷积层都会被使用
            if not self.no_torsion and hasattr(self, 'tor_bond_conv'):
                # 创建虚拟的扭转角数据来触发相关参数
                dummy_bonds = torch.zeros((2, 1), dtype=torch.long, device=self.device)
                dummy_bond_pos = torch.zeros((1, 3), device=self.device)
                dummy_bond_batch = torch.zeros(1, dtype=torch.long, device=self.device)
                dummy_edge_index = torch.zeros((2, 1), dtype=torch.long, device=self.device)
                dummy_edge_vec = torch.zeros((1, 3), device=self.device)
                dummy_edge_attr = self.lig_distance_expansion(dummy_edge_vec.norm(dim=-1))
                dummy_edge_sh = o3.spherical_harmonics(self.sh_irreps, dummy_edge_vec, normalize=True, normalization='component')
                
                dummy_bond_vec = torch.zeros((1, 3), device=self.device) 
                dummy_bond_attr = lig_node_attr.mean(dim=0)[:1] + lig_node_attr.mean(dim=0)[:1]
                dummy_bonds_sh = o3.spherical_harmonics("2e", dummy_bond_vec, normalize=True, normalization='component')
                dummy_tor_edge_sh = self.final_tp_tor(dummy_edge_sh, dummy_bonds_sh[dummy_edge_index[0]])
                dummy_edge_attr = self.final_edge_embedding(dummy_edge_attr)
                dummy_edge_attr = dummy_edge_attr.unsqueeze(0).expand(T, -1, -1)
                dummy_tor_edge_sh = dummy_tor_edge_sh.unsqueeze(0).expand(T, -1, -1)
                dummy_edge_attr = torch.cat([dummy_edge_attr, lig_node_attr[:, dummy_edge_index[1], :self.ns],
                                           dummy_bond_attr.unsqueeze(0).repeat(T, 1, 1)[dummy_edge_index[0]]], -1)
                dummy_tor_pred = self.tor_bond_conv(lig_node_attr, dummy_edge_index.unsqueeze(0).expand(T, -1, -1).reshape(2, -1), 
                                                  dummy_edge_attr, dummy_tor_edge_sh,
                                                  out_nodes=1, reduce='mean')
                dummy_tor_pred = (dummy_tor_pred * time_weights).mean(dim=0)
                dummy_tor_pred = self.tor_final_layer(dummy_tor_pred).squeeze(1)
                
                # 返回一个有梯度的零张量而不是空张量
                tor_pred = dummy_tor_pred * 0.0  # 乘以0确保不影响实际结果，但保持梯度流
            else:
                tor_pred = torch.zeros(1, device=self.device, requires_grad=True)
            
            return tr_pred, rot_pred, tor_pred

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr.mean(dim=0)[tor_bonds[0]] + lig_node_attr.mean(dim=0)[tor_bonds[1]]
        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])
        tor_edge_attr = self.final_edge_embedding(tor_edge_attr)
        tor_edge_index = self.expand_edge_index(tor_edge_index, T)
        tor_edge_attr = tor_edge_attr.unsqueeze(0).expand(T, -1, -1)
        tor_edge_sh = tor_edge_sh.unsqueeze(0).expand(T, -1, -1)
        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[:, tor_edge_index[1], :self.ns],
                                   tor_bond_attr.unsqueeze(0).repeat(T, 1, 1)[tor_edge_index[0]]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                      out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = (tor_pred * time_weights).mean(dim=0)
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        
        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        data['ligand'].node_sigma_emb = self.timestep_emb_func(torch.arange(self.num_timesteps).to(self.device), self.sigma_embed_dim)
        assert data['ligand'].node_sigma_emb.shape == (self.num_timesteps, self.sigma_embed_dim), \
            f"Expected node_sigma_emb shape [{self.num_timesteps}, {self.sigma_embed_dim}], got {data['ligand'].node_sigma_emb.shape}"
        
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)
        edge_sigma_emb = data['ligand'].node_sigma_emb[:, edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], -1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb[0]], -1)
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        data['receptor'].node_sigma_emb = self.timestep_emb_func(torch.arange(self.num_timesteps).to(self.device), self.sigma_embed_dim)
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb[0]], -1)
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[:, edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_atom_conv_graph(self, data):
        data['atom'].node_sigma_emb = self.timestep_emb_func(torch.arange(self.num_timesteps).to(self.device), self.sigma_embed_dim)
        node_attr = torch.cat([data['atom'].x, data['atom'].node_sigma_emb[0]], -1)
        edge_index = data['atom', 'atom'].edge_index
        src, dst = edge_index
        edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['atom'].node_sigma_emb[:, edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], -1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        if torch.is_tensor(lr_cross_distance_cutoff):
            lr_edge_index = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                  data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                  data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            lr_edge_index = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                                  data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        lr_edge_vec = data['receptor'].pos[lr_edge_index[1].long()] - data['ligand'].pos[lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[:, lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], -1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        la_edge_index = radius(data['atom'].pos, data['ligand'].pos, self.lig_max_radius,
                              data['atom'].batch, data['ligand'].batch, max_num_neighbors=10000)
        la_edge_vec = data['atom'].pos[la_edge_index[1].long()] - data['ligand'].pos[la_edge_index[0].long()]
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[:, la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], -1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')

        ar_edge_index = data['atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['atom'].pos[ar_edge_index[0].long()]
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data['atom'].node_sigma_emb[:, ar_edge_index[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], -1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')

        return lr_edge_index, lr_edge_attr, lr_edge_sh, la_edge_index, la_edge_attr, \
               la_edge_sh, ar_edge_index, ar_edge_attr, ar_edge_sh

    def build_center_conv_graph(self, data):
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)
        center_pos = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)
        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[:, edge_index[1].long()]
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
