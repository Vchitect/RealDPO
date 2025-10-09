
## Installation

```
conda create -n real_dpo python=3.10

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Training
cd finetune

需要使用4个GPU训练
sh train_zero_i2v_real_dpo.sh
```

```

## Infer


```bash
# 生成负样本
sh scripts/i2v_reject_sampling

# 训练后推理

# pipe.transformer.load_state_dict(torch.load("../output/real_dpo/mp_rank_00_model_states.pt",map_location='cpu')['module'])
sh scripts/i2v_val_sampling.py

```

## 数据

默认使用demo数据训练，实际训练时需要切换为完整数据

part数据合并
cat RealDPO_part_* > largefile_restored.tar
合并后，可以使用以下命令验证:
tar -tvf largefile_restored.tar