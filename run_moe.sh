set -e
# env
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$PWD/trainer_3m_fix_base:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0"

test_data="heping_all.ark"
config="train_domain_acc_3m_bi_half.yaml"
prior_file="14wh_1434ctc_smooth20.prior"
cmvn_file="global_cmvn.stats.14wh.nodelta"

ext=${test_data##*.}
test_rspec="${ext}:${test_data}"
ii=1
port_id=0
for x in model.epoch-2.step-12000;do
load_path="$x"
model=${load_path##*/}
output_dir=./moe_heping/$model
world_size=`echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/\n/g' | wc -l`
MASTER_PORT=31519
let port_id=$MASTER_PORT+$ii
####MASTER_PORT=31519
MASTER_PORT=$port_id
let ii=$ii+1
####python ./infer_score_enc.py \
python -m torch.distributed.launch --master_port $MASTER_PORT \
    --nproc_per_node $world_size ./infer_score_enc.py \
    --cuda \
    --config $config \
    --load_path $load_path \
    --test_rspec "$test_rspec" \
    --prior_file $prior_file \
    --cmvn_file $cmvn_file \
    --output_dir $output_dir  &
done
wait;

