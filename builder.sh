set -e
# env
export PYTHONPATH=$PWD/trainer_3m_fix/:$PYTHONPATH
export PYTHONPATH=$PWD/TRTAPI++/python/:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/TRTAPI++/build/out/:$LD_LIBRARY_PATH

config="conformer_moe_18l/train_domain_acc_3m_bi_half.yaml"
prior_file="conformer_moe_18l/14wh_1434ctc_smooth20.prior"
cmvn_file="conformer_moe_18l/global_cmvn.stats.14wh.nodelta"
model_file="conformer_moe_18l/model.epoch-2.step-12000"
#model_file="/mnt/leowgyang/0-tasks/220111_conformer_batch_infer/conformer_model/conformer_ad_24l/model.epoch-5.step-1531250"
output_file="conformer_embed_fmoe.plan"

python3 builder.py  \
    --config $config \
    --load_path $model_file \
    --cmvn_file $cmvn_file \
    --output $output_file
