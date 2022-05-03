# wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

nohup python3 train.py --name ants_bees_for_attn --model_type ViT-B_16 --pretrained_dir pretrained/ViT-B_16.npz --num_steps 1000 --gradient_accumulation_steps 8 --warmup_steps 100 --train_batch_size 64 --learning_rate 3e-2 --device cuda:0 --eval_every 100  > ants_bees_ViT-B_16.log 2>&1 & 
