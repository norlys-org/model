
from norlys.baseline import compute_long_term_baseline, read_and_format


df = read_and_format('MAS')
print(df)

start = df.index.min()
end = df.index.max()
baseline = compute_long_term_baseline('MAS', start, end, df)    

# return df[start:end] - baseline[start:end]