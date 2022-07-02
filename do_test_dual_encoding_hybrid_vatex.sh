rootpath=/home/wyb/wyb/workspace/VisualSearch
collectionStrt=single
testCollection=vatex
logger_name=/home/wyb/wyb/workspace/VisualSearch/vatex/cv_tpami_2021/vatex/Full_dtl_back_adv_l1_pooling_catfintune/data_type_google_enc2zh/tri_alpha_0.6/dtl_beta_0.6/l1_gama_1/back_w_0.5/video_layer_1/text_layer_1/text_num_attention_8/text_hidden_size_768/vido_pooling_mean/text_pooling_mean/layer_list_layer.11-layer.10-layer.9-layer.8-layer.7/glr_1e-3/scale_0.001/disc_type_strong/momentum_0.8/optim_adam/dual_encoding_hybrid_concate_full_dp_0.2_measure_cosine_jaccard/visual_feature_i3d_kinetics_visual_rnn_size_512_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-1536_img_0-1536_tag_vocab_size_512/loss_func_mrl_margin_0.2_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0
overwrite=0

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

