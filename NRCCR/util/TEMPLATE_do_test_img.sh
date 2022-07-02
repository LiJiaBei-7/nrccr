rootpath=@@@rootpath@@@
collectionStrt=@@@collectionStrt@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
overwrite=@@@overwrite@@@

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester_img.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name
