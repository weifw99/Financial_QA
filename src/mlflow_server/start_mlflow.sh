

LOG_FILE=log_`date +'%Y%m%d_%H-%M-%S'`.log

nohup mlflow server --host 127.0.0.1 --port 5001 -w 1 --backend-store-uri sqlite:///my.db > ${LOG_FILE} 2>&1 &