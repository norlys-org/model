# model
norlys (new oval representation of nordlys - northern lights) machine learning model.

## Features
The model is split in two parts, it will first compute a score from 0 to 10 based on a set of features ([`app/features`](https://github.com/norlys-org/model/tree/master/app/features)) and averaged with a weighted mean. 
Secondly it will use a pre-trained classifier to establish the current situation (idle, explosion, build-up, ...)

All the data has previously been run through an algorithm that substracts a baseline computed beforehand with the data of the past month.
A third algorithm is used every day to fetch the baseline data.

The model outputs a matrix of points over northern europe with the strength of each point.

## Model training
```bash
python3 train.py
```

## Deployment
The model is deployed on digital ocean's app platform and runs every minute to compute the matrix and store it in cloudfare kv

### Environment variables
- `TGO_PASSWORD` Password the [TGO's magnetogram data](https://flux.phys.uit.no/ascii/).
- `CF_API_TOKEN` Cloudfare KV token

### Data fetching
In order to compute the baseline the model needs a month of historical data. 
Once a day at 2330Z a GitHub action will run the [`retrieve_archive.py`](https://github.com/norlys-org/model/blob/master/retrieve_archive.py) script and push to this same repository, triggering a rebuild by Digital ocean

## Links 

- [Digital Ocean App Platform](https://cloud.digitalocean.com/apps/0477f95d-382c-4e37-a72f-8eeb44943b95/overview?i=67faac)
- [Cloudfare KV](https://dash.cloudflare.com/027c2b0378c6ce9b76e5b5eab615ba04/workers/kv/namespaces/ed64384b958c48eeb86b922b3c1aebb0)
