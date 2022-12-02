rootpath=@@@rootpath@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
overwrite=@@@overwrite@@@

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester_img.py --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name
