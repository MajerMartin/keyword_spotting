#!/bin/sh
# mark the shell script executable by running "chmod +x svm_kernel_all.sh"
# run with "./svm_kernel_all.sh"
SAMPLES_DIR="../data/segmented/"
FEATURES_DIR="../data/features/"
OUTPUT_DIR="../data/distance_matrices/svm/"
OUTPUT_DIR_INVERTED="../data/distance_matrices/svm_inverted/"

### Normal ###

# single speaker without normalization
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10.hdf5 sti_10_10.hdf5 stzcr_10_10.hdf5 ste_sti_stzcr_10_10.hdf5 log_fb_en_25_10_ham.hdf5 log_fb_en_25_10_ham_deltas.hdf5 mfcc_25_10_ham.hdf5 mfcc_25_10_ham_deltas.hdf5 $OUTPUT_DIR single --dist_func True

# all speakers without normalization
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10.hdf5 sti_10_10.hdf5 stzcr_10_10.hdf5 ste_sti_stzcr_10_10.hdf5 log_fb_en_25_10_ham.hdf5 log_fb_en_25_10_ham_deltas.hdf5 mfcc_25_10_ham.hdf5 mfcc_25_10_ham_deltas.hdf5 $OUTPUT_DIR all --dist_func True

# single speaker normalized
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10_norm.hdf5 sti_10_10_norm.hdf5 stzcr_10_10_norm.hdf5 ste_sti_stzcr_10_10_norm.hdf5 log_fb_en_25_10_ham_norm.hdf5 log_fb_en_25_10_ham_deltas_norm.hdf5 mfcc_25_10_ham_norm.hdf5 mfcc_25_10_ham_deltas_norm.hdf5 $OUTPUT_DIR single --dist_func True

# all speakers normalized
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10_norm.hdf5 sti_10_10_norm.hdf5 stzcr_10_10_norm.hdf5 ste_sti_stzcr_10_10_norm.hdf5 log_fb_en_25_10_ham_norm.hdf5 log_fb_en_25_10_ham_deltas_norm.hdf5 mfcc_25_10_ham_norm.hdf5 mfcc_25_10_ham_deltas_norm.hdf5 $OUTPUT_DIR all --dist_func True

# single speaker with custom distance function for ste_sti_stzcr
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim.hdf5 $OUTPUT_DIR single --dist_func True --alpha 0.3 --beta 0.3 --gamma 0.4

# all speakers with custom distance function for ste_sti_stzcr
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim.hdf5 $OUTPUT_DIR all --dist_func True --alpha 0.0 --beta 0.6 --gamma 0.4


### Inverted ###

# single speaker without normalization
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10.hdf5 sti_10_10.hdf5 stzcr_10_10.hdf5 ste_sti_stzcr_10_10.hdf5 log_fb_en_25_10_ham.hdf5 log_fb_en_25_10_ham_deltas.hdf5 mfcc_25_10_ham.hdf5 mfcc_25_10_ham_deltas.hdf5 $OUTPUT_DIR_INVERTED single --inverse True --dist_func True

# all speakers without normalization
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10.hdf5 sti_10_10.hdf5 stzcr_10_10.hdf5 ste_sti_stzcr_10_10.hdf5 log_fb_en_25_10_ham.hdf5 log_fb_en_25_10_ham_deltas.hdf5 mfcc_25_10_ham.hdf5 mfcc_25_10_ham_deltas.hdf5 $OUTPUT_DIR_INVERTED all --inverse True --dist_func True

# single speaker normalized
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10_norm.hdf5 sti_10_10_norm.hdf5 stzcr_10_10_norm.hdf5 ste_sti_stzcr_10_10_norm.hdf5 log_fb_en_25_10_ham_norm.hdf5 log_fb_en_25_10_ham_deltas_norm.hdf5 mfcc_25_10_ham_norm.hdf5 mfcc_25_10_ham_deltas_norm.hdf5 $OUTPUT_DIR_INVERTED single --inverse True --dist_func True

# all speakers normalized
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_10_10_norm.hdf5 sti_10_10_norm.hdf5 stzcr_10_10_norm.hdf5 ste_sti_stzcr_10_10_norm.hdf5 log_fb_en_25_10_ham_norm.hdf5 log_fb_en_25_10_ham_deltas_norm.hdf5 mfcc_25_10_ham_norm.hdf5 mfcc_25_10_ham_deltas_norm.hdf5 $OUTPUT_DIR_INVERTED all --inverse True --dist_func True

# single speaker with custom distance function for ste_sti_stzcr
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim.hdf5 $OUTPUT_DIR_INVERTED single --inverse True --dist_func True --alpha 0.3 --beta 0.3 --gamma 0.4

# all speakers with custom distance function for ste_sti_stzcr
python svm_kernel_cli.py $SAMPLES_DIR $FEATURES_DIR ste_sti_stzcr_10_10_norm_optim.hdf5 $OUTPUT_DIR_INVERTED all --inverse True --dist_func True --alpha 0.0 --beta 0.6 --gamma 0.4


