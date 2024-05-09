from sklearn.ensemble import IsolationForest

def anomaly(df, component):
  isolation_forest = IsolationForest(random_state=42)
  X = df[[component]].values 
  isolation_forest.fit(X)

  return isolation_forest.predict(X)

def rolling_anomalies(df, component):
  """
  Number of anomalies over the past 15 minutes
  Depends on IsolationForest anomalies to be computed
  """

  return (df[f'{component}_anomaly'] == -1).astype(int).rolling('15min').sum()

def rolling_gradient(df, component):
  """
  Mean of the derivative of the past 15 minutes 
  """

  return df[component].diff().rolling('15min').mean()

def deflection(df, component):
  """
  Deflection - difference between maximum and minimum - over the past 45 minutes
  """

  return df[component].rolling('45min').apply(lambda x: x.max() - x.min())

# the order matters due to dependencies
features = {
  'anomaly': anomaly,
  'rolling_anomalies': rolling_anomalies,
  'rolling_gradient': rolling_gradient,
  'deflection': deflection
}

def get_features_column_list(component):
  return [f'{component}_{slug}' for slug in features].append(component)

def apply_features(df):
  for component in ['X', 'Y', 'Z']:
    for feature_slug in features:
      df[f'{component}_{feature_slug}'] = features[feature_slug](df, component)
    
  df.dropna(inplace=True)
  return df