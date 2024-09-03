npev=10 
eval_query_bsz=200 # 500
eval_context_bsz=500
collection=charades  # activitynet
eval_id=eval_fawl
root_path=data/
model_dir=VCMR/2024_09_02_14_40_13-charades-fawl


# training

python eval.py \
--collection $collection \
--eval_id $eval_id \
--root_path $root_path  \
--dset_name $collection \
--model_dir $model_dir \
--npev $npev \
--eval_context_bsz $eval_context_bsz \
--eval_query_bsz $eval_query_bsz