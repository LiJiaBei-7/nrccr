rootpath=/home/wyb/wyb/workspace/VisualSearch
collectionStrt=single
testCollection=msrvtt10kyu
logger_name=$1
overwrite=0

gpu=1

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

