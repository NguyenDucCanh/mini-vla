# Mini-VLA



## Collect data
```
python -m scripts.collect_data \      
  --env-id FrankaKitchen-v1 \
  --output-path data/kitchen_imitation_dataset.npz \
  --episodes-per-task 20 \
  --max-steps 100  
  ```

if you want to use LIBERO:
```
python -m scripts.build_libero_npz \
  --hf-dataset physical-intelligence/libero \
  --split train \
  --output-path data/libero_bc.npz \
  --max-samples -1
  ```

## Train

```
python -m scripts.train_diffusion_vla \
  --dataset-path data/kitchen_imitation_dataset.npz \
  --epochs 20 \
  --batch-size 64 \
  --save-path checkpoints/vla_diffusion.pt \
  --device cpu
```

## Test

```
python -m scripts.test_diffusion_vla \
  --checkpoint checkpoints/vla_diffusion.pt \
  --env-id FrankaKitchen-v1 \
  --episodes 10 \
  --max-steps 150 \
  --instruction "open the microwave" \
  --device cpu
```