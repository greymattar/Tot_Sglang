module use /soft/modulefiles
module load conda

export EXP_ROOT=/lus/grand/projects/VeloC/atkia/tot_fastTree
conda activate "$EXP_ROOT/env/py"

export PYTHONPATH="$EXP_ROOT:$PYTHONPATH"
# HuggingFace cache + token
export HF_HOME=/lus/grand/projects/VeloC/atkia/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export HF_TOKEN="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"

