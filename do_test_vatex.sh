rootpath=$1
collectionStrt=single
testCollection=vatex
logger_name=$2
overwrite=0

gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python tester_vid.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

