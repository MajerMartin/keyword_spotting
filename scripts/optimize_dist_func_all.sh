#!/bin/sh
# mark the shell script executable by running "chmod +x optimize_dist_func_all.sh"
# run with "./optimize_dist_func_all.sh"
SAMPLES_DIR="../data/segmented/"
FEATURES_DIR="../data/features/"
OUTPUT_DIR="../data/params/"

python optimize_dist_func_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim.hdf5 $OUTPUT_DIR single
python optimize_dist_func_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim.hdf5 $OUTPUT_DIR all