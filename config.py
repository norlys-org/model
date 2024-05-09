# Paths
TRAIN_PATH = 'data/train.csv'
SOLAR_WIND_PATH = 'data/sw.csv'
QUANTILES_PATH = 'data/quantiles.json'

# Model
EVENT_WINDOW_OFFSET = 2 # hours
ROLLING_WINDOW_SIZE = 45 # minutes (since it is 1m data)
CLASSES = ['explosion', 'build', 'recovery', 'energy_entry']

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
STATIONS = {
    'NAL': { 'X': 14, 'Y': 13, 'lat': 78.92, 'lon': 11.95, 'slug': 'nal1a', 'source': 'tgo' },
    'LYR': { 'X': 16, 'Y': 15, 'lat': 78.20, 'lon': 15.82, 'slug': 'lyr2a', 'source': 'tgo' },
    # 'HOR': { 'X': 20, 'Y': 17, 'lat': 77.00, 'lon': 15.60 },
    # 'HOP': { 'X': 22, 'Y': 14, 'lat': 76.51, 'lon': 25.01 },
    'BJN': { 'X': 21, 'Y': 15, 'lat': 74.50, 'lon': 19.20, 'slug': 'bjn1a', 'source': 'tgo' },
    'NOR': { 'X': 25, 'Y': 12, 'lat': 71.09, 'lon': 25.79, 'slug': 'nor1a', 'source': 'tgo' },
    'SOR': { 'X': 20, 'Y': 12, 'lat': 70.54, 'lon': 22.22, 'slug': 'sor1a', 'source': 'tgo' },
    'KEV': { 'X': 18, 'Y': 9, 'lat': 69.76, 'lon': 27.01, 'slug': 'kev', 'source': 'fmi' },
    'TRO': { 'X': 20, 'Y': 11, 'lat': 69.66, 'lon': 18.94, 'slug': 'tro2a', 'source': 'tgo' },
    'MAS': { 'X': 18, 'Y': 11, 'lat': 69.46, 'lon': 23.70, 'slug': 'mas', 'source': 'fmi' },
    'AND': { 'X': 20, 'Y': 11, 'lat': 69.30, 'lon': 16.03, 'slug': 'and1a', 'source': 'tgo' },
    'JCK': { 'X': 19, 'Y': 11, 'lat': 69.29, 'lon': 16.04, 'slug': 'jck1a', 'source': 'tgo' },
    'KIL': { 'X': 18, 'Y': 10, 'lat': 69.06, 'lon': 20.77, 'slug': 'kil', 'source': 'fmi' },
    'IVA': { 'X': 16, 'Y': 8, 'lat': 68.56, 'lon': 27.29, 'slug': 'iva', 'source': 'fmi' },
    # 'ABK': { 'X': 17, 'Y': 10, 'lat': 68.35, 'lon': 18.82 },
    # 'LEK': { 'X': 15, 'Y': 9, 'lat': 68.13, 'lon': 13.54 },
    'MUO': { 'X': 15, 'Y': 8, 'lat': 68.02, 'lon': 23.53, 'slug': 'muo', 'source': 'fmi' },
    # 'LOZ': { 'X': 12, 'Y': 6, 'lat': 67.97, 'lon': 35.08 },
    # 'KIR': { 'X': 14, 'Y': 8, 'lat': 67.84, 'lon': 20.42 },
    # 'SOD': { 'X': 13, 'Y': 7, 'lat': 67.37, 'lon': 26.63 },
    'PEL': { 'X': 12, 'Y': 7, 'lat': 66.90, 'lon': 24.08, 'slug': 'pel', 'source': 'fmi' },
    'DON': { 'X': 11, 'Y': 6, 'lat': 66.11, 'lon': 12.50, 'slug': 'don1a', 'source': 'tgo' },
    'RAN': { 'X': 10, 'Y': 5, 'lat': 65.54, 'lon': 26.25, 'slug': 'ran', 'source': 'fmi' },
    'RVK': { 'X': 9, 'Y': 5, 'lat': 64.94, 'lon': 10.98, 'slug': 'rvk1a', 'source': 'tgo' },
    # 'LYC': { 'X': 19, 'Y': 7, 'lat': 64.61o, 'lon': 18.75 },
    'OUJ': { 'X': 7, 'Y': 5, 'lat': 64.52, 'lon': 27.23, 'slug': 'ouj', 'source': 'fmi' },
    'MEK': { 'X': 4, 'Y': 4, 'lat': 62.77, 'lon': 30.97, 'slug': 'mek', 'source': 'fmi' },
    'HAN': { 'X': 4, 'Y': 4, 'lat': 62.25, 'lon': 26.60, 'slug': 'han', 'source': 'fmi' },
    'DOB': { 'X': 5, 'Y': 4, 'lat': 62.07, 'lon': 9.11, 'slug': 'dob1a', 'source': 'tgo' },
    'SOL': { 'X': 4, 'Y': 4, 'lat': 61.08, 'lon': 4.84, 'slug': 'sol1a', 'source': 'tgo' },
    'NUR': { 'X': 5, 'Y': 3, 'lat': 60.50, 'lon': 24.65, 'slug': 'nur', 'source': 'fmi' },
    # 'UPS': { 'X': 4, 'Y': 3, 'lat': 59.90, 'lon': 17.35 },
    'KAR': { 'X': 3, 'Y': 3, 'lat': 59.21, 'lon': 5.24, 'slug': 'kar1a', 'source': 'tgo' },
    'TAR': { 'X': 3, 'Y': 3, 'lat': 58.26, 'lon': 26.46, 'slug': 'tar', 'source': 'fmi' },
}