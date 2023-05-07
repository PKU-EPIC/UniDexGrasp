CUDA_VISIBLE_DEVICES=3,0,1,2,4,5,6,7 \
python train.py \
--task=ShadowHandRandomLoadVision \
--algo=ppo \
--seed=0 \
--rl_device=cuda:1 \
--sim_device=cuda:1 \
--logdir=logs/test \
--headless \
--vision \
--backbone_type pn #pn/transpn 
#--test