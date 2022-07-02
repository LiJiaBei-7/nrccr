rootpath=/home/zms/VisualSearch
collectionStrt=single
testCollection=multi30k
logger_name=$1
overwrite=0

gpu=1

CUDA_VISIBLE_DEVICES=$gpu python tester_img.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

