# BossNAS — Architecture Search (Soft Margin PP)

## 1. Cài đặt môi trường

```bash
pip install torch==2.8.0 torchvision==0.23.0
pip install mmcv==1.0.3
```

---

## 2. Chạy search

```bash
cd ../BossNAS/searching

torchrun \
    --nproc_per_node=1 \
    train.py \
    configs/hytra_pp_bs16_accumulate8_ep6_gpu1_resume.py \
    --work_dir work_dirs/hytra_pp_softmargin
```

---

