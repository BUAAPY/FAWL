collection=charades
visual_feature=i3d_rgb_lgi
root_path=data/

#CF
clip_scale_w=0.4 # 4
frame_scale_w=0.6 # 0.6
# eval
eval_context_bsz=500
eval_query_bsz=500 #500

margin=0.2
intra_margin=0.2 # 0.2

# proposals
max_ctx_l=64 #64
num_gauss_center=16 #16

exp_id=fawl
device_ids=1
warmup_epoch=7
alpha4=1
alpha5=1


# training
python train.py  \
--collection $collection \
--visual_feature $visual_feature \
--root_path $root_path \
--dset_name $collection \
--exp_id $exp_id \
--clip_scale_w $clip_scale_w \
--frame_scale_w $frame_scale_w \
--device_ids $device_ids \
--eval_context_bsz $eval_context_bsz \
--eval_query_bsz $eval_query_bsz \
--margin $margin \
--num_gauss_center $num_gauss_center \
--intra_margin $intra_margin \
--max_ctx_l $max_ctx_l \
--warmup_epoch $warmup_epoch \
--alpha4 $alpha4 \
--alpha5 $alpha5