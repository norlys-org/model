from norlys import model
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logging.info('Starting model training process.')
model.train_0m_classifier()