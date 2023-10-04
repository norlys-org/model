from norlys.data_utils import get_clipped_data
from norlys.features.scores import compute_scores

df = get_clipped_data()
print(compute_scores(df))