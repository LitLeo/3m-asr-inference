set -e
# env
export PYTHONPATH=$PWD/TRTAPI++/python/:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/TRTAPI++/build/out/:$LD_LIBRARY_PATH

python3 infer.py -p conformer_embed_fmoe.plan -i conformer_moe_18l/case/feat.npy
