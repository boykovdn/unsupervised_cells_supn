python gridsearch_experiment.py \
	--L1_REG_WEIGHT 0 \
	--DEPTH 2 3 4 5 6\
	--ENCODING_DIMENSION 100 \
	--LEARNING_RATE 0.001 \
	--FIXED_VAR 0.09 \
	--local_minimum_detection_threshold 0.00000001 \
	--experiment_tag bf_depth_test \
	--common_params_yaml gridsearch_vae.yaml
