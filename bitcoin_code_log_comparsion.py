import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import kagglehub

# ==========================================
# 1. 基础组件 (保持不变)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        self.eps = eps

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias
        else:
            return (x - self.affine_bias) / (self.affine_weight + self.eps) * self.stdev + self.mean

class TemporalDecayPooling(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.weights = nn.Parameter(torch.linspace(-1, 0, seq_len))

    def forward(self, x):
        norm_weights = torch.softmax(self.weights, dim=0)
        return torch.sum(x * norm_weights.unsqueeze(-1).unsqueeze(0), dim=1)

# ==========================================
# 2. 核心架构: Alpha-Focus Single-Stick (Baseline)
# ==========================================
class AlphaFocus_SingleStick(nn.Module):
    def __init__(self, seq_len=96, num_vars=9, embed_dim=128):
        super().__init__()
        self.num_vars = num_vars
        self.revin = RevIN(num_vars)
        
        # 🥢 单筷子投影: 只接受原始价格序列 (1路输入)
        # 对比 V3 (3路) 和 V4 (4路)
        self.single_proj = nn.Linear(1, embed_dim)

        # Stage 1: 共享时间编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*2, 
            batch_first=True, norm_first=True, dropout=0.1
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.time_pooling = TemporalDecayPooling(seq_len)

        # Stage 2: iTransformer 变量间博弈
        var_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512, 
            batch_first=True, norm_first=True, dropout=0.1
        )
        self.var_attn = nn.TransformerEncoder(var_layer, num_layers=3)
        
        self.head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.final_gate = nn.Linear(num_vars, 1)

    def forward(self, x):
        B, L, N = x.shape
        x = self.revin(x, 'norm')
        
        # 🥢 单筷子逻辑: 直接升维，没有任何对抗或动量补充
        feat = x.unsqueeze(-1) # [B, L, N, 1]
        
        # [B, L, N, 1] -> [B, N, L, D]
        feat = self.single_proj(feat).permute(0, 2, 1, 3) 
        
        # 2. 共享时间编码 (与 V3 一致)
        feat = feat.reshape(B * N, L, -1)
        feat = self.temporal_encoder(feat)
        feat = self.time_pooling(feat) # [B*N, D]
        feat = feat.reshape(B, N, -1) 
        
        # 3. 变量间博弈
        feat = self.var_attn(feat)
        
        # 4. 最终决策
        out_vars = self.head(feat).squeeze(-1) 
        return self.final_gate(out_vars)

# ==========================================
# 3. 数据与训练 (保持完全一致)
# ==========================================
class BTCDatasetExpert(Dataset):
    def __init__(self, csv_path, flag='train'):
        df = pd.read_csv(csv_path)
        df.columns = [c.replace('"', '').strip().lower() for c in df.columns]
        if len(df) > 500000:
            df = df.iloc[-500000:].reset_index(drop=True)
        close = df['close'] if 'close' in df.columns else df.iloc[:, 3]
        delta = close.diff()
        df['f1'] = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(14).mean()/(delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-5))) 
        df['f2'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['f3'] = close.pct_change().rolling(20).std()
        df = df.fillna(0)
        target_cols = ['open','high','low','close','volume','quote_asset_volume','f1','f2','f3']
        existing_cols = [c for c in target_cols if c in df.columns]
        while len(existing_cols) < 9: existing_cols.append('close')
        data = df[existing_cols[:9]].values.astype(np.float32)
        data = (data - data.mean(0)) / (data.std(0) + 1e-5)
        labels = (close.shift(-1) > close).astype(float).values
        data = data[30:-1]; labels = labels[30:-1]
        split = int(len(data) * 0.8)
        self.data, self.labels = (data[:split], labels[:split]) if flag == 'train' else (data[split:], labels[split:])

    def __getitem__(self, i):
        return torch.tensor(self.data[i:i+96]), torch.tensor(self.labels[i+95]).unsqueeze(-1)
    def __len__(self): return len(self.data) - 100

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    if local_rank == 0: print("📥 Single-Stick Baseline: Preparing data...")
    dist.barrier()
    
    path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
    real_csv = os.path.join(path, [f for f in os.listdir(path) if f.endswith('.csv')][0])

    train_ds = BTCDatasetExpert(real_csv, 'train')
    val_ds = BTCDatasetExpert(real_csv, 'val')
    train_loader = DataLoader(train_ds, batch_size=256, sampler=DistributedSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=256)

    model = DDP(AlphaFocus_SingleStick(num_vars=9).to(local_rank), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    for epoch in range(20): # 跑 10 个 Epoch 足以看出差距
        train_loader.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, disable=(local_rank != 0))
        for bx, by in pbar:
            optimizer.zero_grad()
            bx, by = bx.to(local_rank), by.to(local_rank)
            with autocast('cuda'):
                loss = criterion(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        model.eval()
        v_c, v_t = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                with autocast('cuda'): logits = model(bx.to(local_rank))
                v_c += ((torch.sigmoid(logits) > 0.5).float() == by.to(local_rank)).sum().item()
                v_t += by.size(0)
        
        v_c_tensor = torch.tensor(v_c).to(local_rank); v_t_tensor = torch.tensor(v_t).to(local_rank)
        dist.all_reduce(v_c_tensor); dist.all_reduce(v_t_tensor)
        
        if local_rank == 0:
            print(f"📉 Single-Stick Epoch {epoch+1} Val Acc: {v_c_tensor.item()/v_t_tensor.item():.2%}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()


(base) root@ubuntu22:~# torchrun --nproc_per_node=4 bitcoin2.py
W0219 03:41:12.143000 6209 site-packages/torch/distributed/run.py:793] 
W0219 03:41:12.143000 6209 site-packages/torch/distributed/run.py:793] *****************************************
W0219 03:41:12.143000 6209 site-packages/torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0219 03:41:12.143000 6209 site-packages/torch/distributed/run.py:793] *****************************************
📥 Single-Stick Baseline: Preparing data...
[rank2]:[W219 03:41:15.167016385 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank3]:[W219 03:41:15.167015268 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 3]  using GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank1]:[W219 03:41:15.170426440 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank0]:[W219 03:41:15.214880633 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
/root/anaconda3/lib/python3.11/site-packages/torch/nn/modules/transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/root/anaconda3/lib/python3.11/site-packages/torch/nn/modules/transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/root/anaconda3/lib/python3.11/site-packages/torch/nn/modules/transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/root/anaconda3/lib/python3.11/site-packages/torch/nn/modules/transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 15.92it/s]
📉 Single-Stick Epoch 1 Val Acc: 53.52%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:22<00:00, 17.01it/s]
📉 Single-Stick Epoch 2 Val Acc: 53.50%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.62it/s]
📉 Single-Stick Epoch 3 Val Acc: 53.53%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 15.97it/s]
📉 Single-Stick Epoch 4 Val Acc: 53.49%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 15.99it/s]
📉 Single-Stick Epoch 5 Val Acc: 53.53%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.36it/s]
📉 Single-Stick Epoch 6 Val Acc: 53.52%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.73it/s]
📉 Single-Stick Epoch 7 Val Acc: 53.54%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.59it/s]
📉 Single-Stick Epoch 8 Val Acc: 53.64%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 16.08it/s]
📉 Single-Stick Epoch 9 Val Acc: 53.48%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.49it/s]
📉 Single-Stick Epoch 10 Val Acc: 53.54%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.58it/s]
📉 Single-Stick Epoch 11 Val Acc: 53.53%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.50it/s]
📉 Single-Stick Epoch 12 Val Acc: 53.47%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 16.05it/s]
📉 Single-Stick Epoch 13 Val Acc: 53.54%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 16.26it/s]
📉 Single-Stick Epoch 14 Val Acc: 53.35%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.64it/s]
📉 Single-Stick Epoch 15 Val Acc: 53.54%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.66it/s]
📉 Single-Stick Epoch 16 Val Acc: 53.53%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 16.25it/s]
📉 Single-Stick Epoch 17 Val Acc: 53.52%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:24<00:00, 16.15it/s]
📉 Single-Stick Epoch 18 Val Acc: 53.54%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.87it/s]
📉 Single-Stick Epoch 19 Val Acc: 53.54%
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:23<00:00, 16.65it/s]
📉 Single-Stick Epoch 20 Val Acc: 53.55%
(base) root@ubuntu22:~# 