# Features needed
# - deflection score for explosion label, only if >= 5 data points. percentile of deflection for 50%, 60%, 70%, 80% and 90%
# - isolation forest number of anomalies + percentile of number of anomalies in event
# - "patterns" on the magnetometre circular map for each class, e.g. arc for "build-up", covers the whole sky for "explosion"
# - general "disturbance" score based on the deflection score and the anomaly score + the class
#
# Features to be added to the model
# - length of build-up if one occured in the past 45 minutes
# - position of arc depending on the derivative of Z