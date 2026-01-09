import logging
import os
from datetime import datetime

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
log_filename = f"{timestamp}.log"
log_file = os.path.join(logs_dir, log_filename)

logging.basicConfig(
    filename=log_file,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)





