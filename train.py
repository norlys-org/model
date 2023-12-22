from norlys import model
import logging

from norlys.features.quantiles import save_quantiles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logging.info('Compute and save quantiles.')
save_quantiles()

logging.info('Starting model training process.')
model.train_0m_classifier()