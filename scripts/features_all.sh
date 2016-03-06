#!/bin/sh
# mark the shell script executable by running "chmod +x features_all.sh"
# run with "./features_all.sh"
INPUT_DIR="../data/segmented"
OUTPUT_DIR="../data/features/"

# without normalization
python features_cli.py ste $INPUT_DIR $OUTPUT_DIR"ste_10_10.hdf5" 0.01 0.01
python features_cli.py sti $INPUT_DIR $OUTPUT_DIR"sti_10_10.hdf5" 0.01 0.01
python features_cli.py stzcr $INPUT_DIR $OUTPUT_DIR"stzcr_10_10.hdf5" 0.01 0.01
python features_cli.py ste_sti_stzcr $INPUT_DIR $OUTPUT_DIR"ste_sti_stzcr_10_10.hdf5" 0.01 0.01
python features_cli.py log_fb_en --filter_count 40 $INPUT_DIR $OUTPUT_DIR"log_fb_en_25_10_ham.hdf5" 0.025 0.01 --win_func hamming
python features_cli.py log_fb_en --filter_count 40 --deltas True --ddeltas True $INPUT_DIR $OUTPUT_DIR"log_fb_en_25_10_ham_deltas.hdf5" 0.025 0.01 --win_func hamming
python features_cli.py mfcc $INPUT_DIR $OUTPUT_DIR"mfcc_25_10_ham.hdf5" 0.025 0.01 --win_func hamming
python features_cli.py mfcc --deltas True --ddeltas True $INPUT_DIR $OUTPUT_DIR"mfcc_25_10_ham_deltas.hdf5" 0.025 0.01 --win_func hamming

# normalized
python features_cli.py ste $INPUT_DIR $OUTPUT_DIR"ste_10_10_norm.hdf5" 0.01 0.01 --norm True
python features_cli.py sti $INPUT_DIR $OUTPUT_DIR"sti_10_10_norm.hdf5" 0.01 0.01 --norm True
python features_cli.py stzcr $INPUT_DIR $OUTPUT_DIR"stzcr_10_10_norm.hdf5" 0.01 0.01 --norm True
python features_cli.py ste_sti_stzcr $INPUT_DIR $OUTPUT_DIR"ste_sti_stzcr_10_10_norm.hdf5" 0.01 0.01 --norm True
python features_cli.py log_fb_en --filter_count 40 $INPUT_DIR $OUTPUT_DIR"log_fb_en_25_10_ham_norm.hdf5" 0.025 0.01 --win_func hamming --norm True
python features_cli.py log_fb_en --filter_count 40 --deltas True --ddeltas True $INPUT_DIR $OUTPUT_DIR"log_fb_en_25_10_ham_deltas_norm.hdf5" 0.025 0.01 --win_func hamming --norm True
python features_cli.py mfcc $INPUT_DIR $OUTPUT_DIR"mfcc_25_10_ham_norm.hdf5" 0.025 0.01 --win_func hamming --norm True
python features_cli.py mfcc --deltas True --ddeltas True $INPUT_DIR $OUTPUT_DIR"mfcc_25_10_ham_deltas_norm.hdf5" 0.025 0.01 --win_func hamming --norm True

# copies for optimization
python features_cli.py ste_sti_stzcr $INPUT_DIR $OUTPUT_DIR"ste_sti_stzcr_10_10_norm_optim.hdf5" 0.01 0.01 --norm True
python features_cli.py ste_sti_stzcr $INPUT_DIR $OUTPUT_DIR"ste_sti_stzcr_10_10_norm_optim_single.hdf5" 0.01 0.01 --norm True
python features_cli.py ste_sti_stzcr $INPUT_DIR $OUTPUT_DIR"ste_sti_stzcr_10_10_norm_optim_all.hdf5" 0.01 0.01 --norm True