import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("dealmonitor_ml.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dealmonitor_ml")