#!/bin/sh
# mark the shell script executable by running "chmod +x partial_dist_matrix_all.sh"
# run with "./partial_dist_matrix_all.sh"
SAMPLES_DIR="../data/segmented/"
FEATURES_DIR="../data/features/"
OUTPUT_DIR="../data/distance_matrices/dtw/"

# without normalization
python partial_dist_matrix_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10.hdf5 sti_10_10.hdf5 stzcr_10_10.hdf5 ste_sti_stzcr_10_10.hdf5 log_fb_en_25_10_ham.hdf5 log_fb_en_25_10_ham_deltas.hdf5 mfcc_25_10_ham.hdf5 mfcc_25_10_ham_deltas.hdf5 $OUTPUT_DIR --dist_func True

#  normalized
python partial_dist_matrix_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10_norm.hdf5 sti_10_10_norm.hdf5 stzcr_10_10_norm.hdf5 ste_sti_stzcr_10_10_norm.hdf5 log_fb_en_25_10_ham_norm.hdf5 log_fb_en_25_10_ham_deltas_norm.hdf5 mfcc_25_10_ham_norm.hdf5 mfcc_25_10_ham_deltas_norm.hdf5 $OUTPUT_DIR --dist_func True

# custom distance function for ste_sti_stzcr
python partial_dist_matrix_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim_single.hdf5 $OUTPUT_DIR --dist_func True --alpha 0.3 --beta 0.3 --gamma 0.4
python partial_dist_matrix_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim_all.hdf5 $OUTPUT_DIR --dist_func True --alpha 0.0 --beta 0.6 --gamma 0.4

# bottleneck features
python partial_dist_matrix_cli.py $SAMPLES_DIR $FEATURES_DIR BN3bip_MM_8.hdf5 BN3bip_MM_16a.hdf5 BN3bip_MM-SA_8.hdf5 BN3bip_MM-SA_16.hdf5 BN3bip_MM-SA_32.hdf5 $OUTPUT_DIR
