python gridsearch_experiment.py \
	--L1_REG_WEIGHT 0 \
	--DEPTH 5\
	--ENCODING_DIMENSION 20 100 200 300 500 1000 \
	--LEARNING_RATE 0.001 \
	--FIXED_VAR 0.06 \
	--local_minimum_detection_threshold 0.00000001 \
	--experiment_tag logl_peak \
	--common_params_yaml gridsearch_vae.yaml
