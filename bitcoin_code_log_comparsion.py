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


itransformer,20 epoch
-----------------------------------------------------------------------------------------

import os
import math
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
# 1. iTransformer 官方核心底层组件
# ==========================================
class InvertedEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        # 这里的 c_in 是序列长度 (96)，d_model 是特征维度
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, L, N] -> [B, N, L] 
        x = x.permute(0, 2, 1)
        return self.dropout(self.value_embedding(x))

class iTransformer_EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        new_x, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(new_x))
        # Feed Forward
        y = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

# ==========================================
# 2. 核心架构: Alpha-Focus iTransformer
# ==========================================
class AlphaFocus_iTransformer(nn.Module):
    def __init__(self, seq_len=96, num_vars=9, d_model=128, n_heads=8, e_layers=3):
        super().__init__()
        self.revin = RevIN(num_vars)
        
        # 官方 Inverted Embedding: 将 L(96) 映射到 d_model
        self.enc_embedding = InvertedEmbedding(seq_len, d_model)
        
        # 堆叠 Encoder Layers
        self.encoder = nn.ModuleList([
            iTransformer_EncoderLayer(d_model, n_heads, d_ff=d_model*4) 
            for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        # 最终分类头: 全局特征 -> 二分类
        self.projection = nn.Linear(d_model * num_vars, 1)

    def forward(self, x):
        B, L, N = x.shape
        x = self.revin(x.to(torch.float32), 'norm')
        
        # 1. 序列倒置 Embedding [B, N, d_model]
        enc_out = self.enc_embedding(x)
        
        # 2. 变量间注意力交互
        for layer in self.encoder:
            enc_out = layer(enc_out)
        enc_out = self.norm(enc_out)
        
        # 3. 展平并输出 [B, N*d_model] -> [B, 1]
        return self.projection(enc_out.reshape(B, -1))

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
        return (x - self.affine_bias) / (self.affine_weight + self.eps) * self.stdev + self.mean

# ==========================================
# 3. 数据集 (完全对齐你的 0.5M 逻辑)
# ==========================================
class BTCDatasetExpert(Dataset):
    def __init__(self, csv_path, flag='train'):
        df = pd.read_csv(csv_path)
        df.columns = [c.replace('"', '').strip().lower() for c in df.columns]
        if len(df) > 500000:
            df = df.iloc[-500000:].reset_index(drop=True)
            
        def find_c(aliases):
            for a in aliases:
                if a in df.columns: return df[a]
            return df.iloc[:, 3]

        close = find_c(['close', 'price_close'])
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
        labels = (close.shift(-1) > close).astype(np.float32).values
        
        data = data[30:-1]; labels = labels[30:-1]
        split = int(len(data) * 0.8)
        self.data, self.labels = (data[:split], labels[:split]) if flag == 'train' else (data[split:], labels[split:])

    def __getitem__(self, i):
        return torch.tensor(self.data[i:i+96]), torch.tensor(self.labels[i+95]).unsqueeze(-1)
    def __len__(self): return len(self.data) - 100

# ==========================================
# 4. 分布式引擎
# ==========================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    if local_rank == 0: print("🔥 Official iTransformer Engine (Binary Classification) Launching...")
    
    path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
    real_csv = os.path.join(path, [f for f in os.listdir(path) if f.endswith('.csv')][0])

    train_ds = BTCDatasetExpert(real_csv, 'train')
    val_ds = BTCDatasetExpert(real_csv, 'val')
    train_loader = DataLoader(train_ds, batch_size=256, sampler=DistributedSampler(train_ds), num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = DDP(AlphaFocus_iTransformer().to(local_rank), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion, scaler = nn.BCEWithLogitsLoss(), GradScaler('cuda')

    for epoch in range(20):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, disable=(local_rank != 0))
        for bx, by in pbar:
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(bx.to(local_rank))
                loss = criterion(logits, by.to(local_rank))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        model.eval(); v_c, v_t = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                with autocast('cuda'): logits = model(bx.to(local_rank))
                v_c += ((torch.sigmoid(logits) > 0.5).float() == by.to(local_rank)).sum().item()
                v_t += by.size(0)
        
        v_c_tensor = torch.tensor(v_c).to(local_rank); v_t_tensor = torch.tensor(v_t).to(local_rank)
        dist.all_reduce(v_c_tensor); dist.all_reduce(v_t_tensor)
        
        if local_rank == 0:
            print(f"📉 iTransformer Epoch {epoch+1} Val Acc: {v_c_tensor.item()/v_t_tensor.item():.2%}")

    dist.destroy_process_group()

if __name__ == "__main__": main()

----------------------------------------
 Official iTransformer Engine (Binary Classification) Launching...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 60.22it/s]
📉 iTransformer Epoch 1 Val Acc: 52.56%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 79.91it/s]
📉 iTransformer Epoch 2 Val Acc: 52.97%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 82.56it/s]
📉 iTransformer Epoch 3 Val Acc: 53.08%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 83.00it/s]
📉 iTransformer Epoch 4 Val Acc: 53.20%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 77.96it/s]
📉 iTransformer Epoch 5 Val Acc: 52.29%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 81.12it/s]
📉 iTransformer Epoch 6 Val Acc: 53.07%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 77.37it/s]
📉 iTransformer Epoch 7 Val Acc: 52.66%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 83.07it/s]
📉 iTransformer Epoch 8 Val Acc: 52.78%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 81.54it/s]
📉 iTransformer Epoch 9 Val Acc: 53.13%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 75.55it/s]
📉 iTransformer Epoch 10 Val Acc: 53.09%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 79.89it/s]
📉 iTransformer Epoch 11 Val Acc: 52.85%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 75.30it/s]
📉 iTransformer Epoch 12 Val Acc: 53.38%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 82.15it/s]
📉 iTransformer Epoch 13 Val Acc: 53.06%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 82.80it/s]
📉 iTransformer Epoch 14 Val Acc: 53.03%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 82.65it/s]
📉 iTransformer Epoch 15 Val Acc: 53.11%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 75.72it/s]
📉 iTransformer Epoch 16 Val Acc: 52.73%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 71.77it/s]
📉 iTransformer Epoch 17 Val Acc: 53.16%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 75.40it/s]
📉 iTransformer Epoch 18 Val Acc: 52.93%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 73.17it/s]
📉 iTransformer Epoch 19 Val Acc: 52.96%
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:04<00:00, 82.69it/s]
📉 iTransformer Epoch 20 Val Acc: 52.97%
(base) root@ubuntu22:~# 

patch-tst， 20 epoch, has some dataleak problem
-------------------------------------------------------------------------------------------
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
# 1. PatchTST 核心组件
# ==========================================
class PatchTST_Backbone(nn.Module):
    def __init__(self, seq_len=96, patch_len=16, stride=8, d_model=128, n_heads=8, e_layers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (max(seq_len, patch_len) - patch_len) // stride + 1
        
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            batch_first=True, norm_first=False, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*N, L, 1] -> [B*N, num_patches, patch_len]
        x = x.squeeze(-1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.value_embedding(x) 
        x = x + self.pos_embedding
        x = self.dropout(x)
        return self.encoder(x)

class AlphaFocus_PatchTST(nn.Module):
    def __init__(self, num_vars=9, seq_len=96, d_model=128):
        super().__init__()
        self.revin = RevIN(num_vars)
        self.backbone = PatchTST_Backbone(seq_len=seq_len, d_model=d_model)
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(d_model * self.backbone.num_patches, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.final_gate = nn.Linear(num_vars, 1)

    def forward(self, x):
        B, L, N = x.shape
        x = self.revin(x.to(torch.float32), 'norm')
        
        # 核心：Channel Independence 处理
        x = x.permute(0, 2, 1).reshape(B * N, L, 1)
        feat = self.backbone(x) # [B*N, num_patches, d_model]
        
        out = self.head(feat).reshape(B, N)
        return self.final_gate(out)

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
        return (x - self.affine_bias) / (self.affine_weight + self.eps) * self.stdev + self.mean

# ==========================================
# 2. 数据处理: 修复 8 vs 9 维度错误
# ==========================================
class BTCDatasetExpert(Dataset):
    def __init__(self, csv_path, flag='train'):
        df = pd.read_csv(csv_path)
        df.columns = [c.replace('"', '').strip().lower() for c in df.columns]
        if len(df) > 500000:
            df = df.iloc[-500000:].reset_index(drop=True)
            
        def find_c(aliases):
            for a in aliases:
                if a in df.columns: return df[a]
            return df.iloc[:, 3]

        close = find_c(['close', 'price_close'])
        delta = close.diff()
        df['f1'] = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(14).mean()/(delta.where(delta < 0, 0).abs().rolling(14).mean() + 1e-5))) 
        df['f2'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['f3'] = close.pct_change().rolling(20).std()
        df = df.fillna(0)
        
        # 🌟 修复点：强制补齐到 9 列，防止 RevIN 报错
        target_cols = ['open','high','low','close','volume','quote_asset_volume','f1','f2','f3']
        existing_cols = [c for c in target_cols if c in df.columns]
        while len(existing_cols) < 9: 
            existing_cols.append('close') # 如果缺列，用收盘价补位
        
        data = df[existing_cols[:9]].values.astype(np.float32)
        data = (data - data.mean(0)) / (data.std(0) + 1e-5)
        labels = (close.shift(-1) > close).astype(np.float32).values
        
        data = data[30:-1]; labels = labels[30:-1]
        split = int(len(data) * 0.8)
        self.data, self.labels = (data[:split], labels[:split]) if flag == 'train' else (data[split:], labels[split:])

    def __getitem__(self, i):
        return torch.tensor(self.data[i:i+96]), torch.tensor(self.labels[i+95]).unsqueeze(-1)
    def __len__(self): return len(self.data) - 100

# ==========================================
# 3. DDP 训练引擎 (带最优模型保存)
# ==========================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    if local_rank == 0: print("🧩 Official PatchTST Engine: Fixed 9-Var Mode")
    
    path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
    real_csv = os.path.join(path, [f for f in os.listdir(path) if f.endswith('.csv')][0])

    train_ds = BTCDatasetExpert(real_csv, 'train')
    val_ds = BTCDatasetExpert(real_csv, 'val')
    train_loader = DataLoader(train_ds, batch_size=256, sampler=DistributedSampler(train_ds), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=4)

    model = DDP(AlphaFocus_PatchTST(num_vars=9).to(local_rank), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion, scaler = nn.BCEWithLogitsLoss(), GradScaler('cuda')
    
    best_acc = 0.0

    for epoch in range(20):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, disable=(local_rank != 0))
        for bx, by in pbar:
            optimizer.zero_grad()
            with autocast('cuda'):
                loss = criterion(model(bx.to(local_rank)), by.to(local_rank))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        model.eval(); v_c, v_t = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                with autocast('cuda'): logits = model(bx.to(local_rank))
                v_c += ((torch.sigmoid(logits) > 0.5).float() == by.to(local_rank)).sum().item()
                v_t += by.size(0)
        
        v_c_r = torch.tensor(v_c).to(local_rank); v_t_r = torch.tensor(v_t).to(local_rank)
        dist.all_reduce(v_c_r); dist.all_reduce(v_t_r)
        
        if local_rank == 0:
            acc = v_c_r.item()/v_t_r.item()
            print(f"📉 Epoch {epoch+1} Val Acc: {acc:.2%}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.module.state_dict(), "patch_tst_best_53.pth")

    dist.destroy_process_group()

if __name__ == "__main__": main()



(base) root@ubuntu22:~# torchrun --nproc_per_node=4 bitcoin_patch_tst.py
W0219 06:33:53.154000 16601 site-packages/torch/distributed/run.py:793] 
W0219 06:33:53.154000 16601 site-packages/torch/distributed/run.py:793] *****************************************
W0219 06:33:53.154000 16601 site-packages/torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0219 06:33:53.154000 16601 site-packages/torch/distributed/run.py:793] *****************************************
🧩 Official PatchTST Engine: Fixed 9-Var Mode
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 59.03it/s]
📉 Epoch 1 Val Acc: 53.53%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.16it/s]
📉 Epoch 2 Val Acc: 53.46%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 61.73it/s]
📉 Epoch 3 Val Acc: 53.53%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 65.21it/s]
📉 Epoch 4 Val Acc: 53.41%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.62it/s]
📉 Epoch 5 Val Acc: 53.32%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 63.07it/s]
📉 Epoch 6 Val Acc: 53.45%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 65.44it/s]
📉 Epoch 7 Val Acc: 53.50%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.47it/s]
📉 Epoch 8 Val Acc: 53.04%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 62.78it/s]
📉 Epoch 9 Val Acc: 53.32%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 65.76it/s]
📉 Epoch 10 Val Acc: 53.43%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.99it/s]
📉 Epoch 11 Val Acc: 53.14%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 63.85it/s]
📉 Epoch 12 Val Acc: 53.38%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.90it/s]
📉 Epoch 13 Val Acc: 53.31%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.71it/s]
📉 Epoch 14 Val Acc: 52.99%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.78it/s]
📉 Epoch 15 Val Acc: 53.10%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 65.63it/s]
📉 Epoch 16 Val Acc: 53.03%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.11it/s]
📉 Epoch 17 Val Acc: 53.30%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 65.58it/s]
📉 Epoch 18 Val Acc: 52.68%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:05<00:00, 65.24it/s]
📉 Epoch 19 Val Acc: 52.56%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:06<00:00, 64.18it/s]
📉 Epoch 20 Val Acc: 52.80%
