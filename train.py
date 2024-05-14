from norlys import model
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # TODO

from norlys.features.quantiles import save_quantiles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

if __name__ == "__main__":
  logging.info('Compute and save quantiles.')
  save_quantiles()

  logging.info('Starting model training process.')
  model.train_0m_classifier()