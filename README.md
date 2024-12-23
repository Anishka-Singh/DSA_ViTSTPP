
```srun --container-mounts=/home:/home --container-workdir=/home/ansingh --container-image=/netscratch/ansingh/ckconv/ckconv.sqsh --ntasks=1 --mem=80GB --partition=RTXA6000 --gpus=1 --time=3:00:00 --immediate=100 python /home/ansingh/Nuwa_Hawkes/main_copy.py --mode hawkes --batch_size 16 --val_batch_size 16 --nuwa_checkpoint /home/ansingh/Nuwa_Hawkes/output/simvp_nighttime_mask/Debug/checkpoint.pth --hawkes_epochs 10 --hawkes_lr 0.001
```

### Train NuwaDynamics
```python main_copy.py \
    --mode nuwa \
    --ex_name nuwa_training \
    --batch_size 64 \
    --val_batch_size 64 \
    --epochs 10 \
    --lr 0.00001
```
### Train ConvHawkes
```python /home/ansingh/Nuwa_Hawkes/main_copy.py --mode hawkes --batch_size 8 --val_batch_size 8 --nuwa_checkpoint /home/ansingh/Nuwa_Hawkes/output/simvp_nighttime_mask/Debug/checkpoints/checkpoint.pth --hawkes_epochs 10 --hawkes_lr 0.001
```
### Train entire pipeline
```python main_copy.py \
    --mode both \
    --ex_name full_pipeline \
    --batch_size 64 \
    --val_batch_size 64 \
    --epochs 10 \
    --lr 0.00001 \
    --hawkes_epochs 30 \
    --hawkes_lr 0.001
```
