rootpath=$1
collectionStrt=single
testCollection=multi30k
logger_name=$2
overwrite=0

gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

