import logging
from pathlib import Path
from datetime import datetime


logs_dir = Path.cwd() / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

# Log file name
log_file = logs_dir / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logging.basicConfig(
    filename=log_file,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)




