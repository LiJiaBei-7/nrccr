rootpath=$1
overwrite=1
collection=mscoco
visual_feature=resnet_152
learning_rate=1e-4
text_hidden_size=768
video_hidden_size=1024
text_num_attention=8
video_num_attention=8
video_pooling=mean
text_pooling=mean
video_layer=1
text_layer=1
frozen=frozen
data_type=google_enc2ja #google_enc2zh
glr=1e-3
scale=0.001
disc_type=not-so-weak # strong # weak
momentum=0.8
optim=adam
tri_alpha=0.6
dtl_beta=0.4
l1_gama=0.1
back_w=0.5
model_type=img
layer_list=layer.11-layer.10-layer.9-layer.8-layer.7-layer.6-layer.5-layer.4-layer.3
framework=Full_mscoco_$frozen/data_type_$data_type/tri_alpha_$tri_alpha/dtl_beta_$dtl_beta/l1_gama_$l1_gama/back_w_$back_w/video_layer_$video_layer/text_layer_$text_layer/text_num_attention_$text_num_attention/text_hidden_size_$text_hidden_size/vido_pooling_$video_pooling/text_pooling_$text_pooling\
/layer_list_$layer_list/glr_$glr/scale_$scale/disc_type_$disc_type/momentum_$momentum/optim_$optim

img_path=mscoco/all_pics
img_encoder=clip
img_encoder_name=ViT-B/32
img_encoder_input_dim=512
batch_size=128

# training
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu python trainer_img.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature\
                                            --framework $framework --learning_rate $learning_rate --frozen $frozen\
                                            --text_hidden_size $text_hidden_size --text_num_attention $text_num_attention\
                                            --video_hidden_size $video_hidden_size --video_num_attention $video_num_attention\
                                            --video_pooling $video_pooling --text_pooling $text_pooling --video_layer $video_layer\
                                            --text_layer $text_layer --layer_list $layer_list --data_type $data_type\
                                            --glr $glr --scale $scale --disc_type $disc_type --momentum $momentum \
                                            --optim $optim --tri_alpha $tri_alpha --dtl_beta $dtl_beta --l1_gama $l1_gama \
                                            --back_w $back_w --img_encoder_name $img_encoder_name --model_type $model_type\
                                            --img_encoder_input_dim $img_encoder_input_dim --batch_size $batch_size \
                                            --img_path $img_path --img_encoder $img_encoder

