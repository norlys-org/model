from norlys.data_utils import get_clipped_data
from norlys.features.deflection import deflection_score

df = get_clipped_data()
print(deflection_score(df))