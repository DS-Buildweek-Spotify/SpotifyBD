import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
from pydantic import BaseModel, confloat

description = """
Deploys a K Nearest Neighbor Model fit on the [Spotify](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=tracks.csv) dataset.

<img src="https://www.bing.com/images/search?view=detailV2&ccid=MyKc5S3r&id=06AF71454D84A89FDBEF3C0BC8F046FF4842E5E5&thid=OIP.MyKc5S3r-n8ER2kwZUHh7gHaDW&mediaurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR33229ce52debfa7f044769306541e1ee%3frik%3d5eVCSP9G8MgLPA%26riu%3dhttp%253a%252f%252fmedia.idownloadblog.com%252fwp-content%252fuploads%252f2016%252f06%252fSpotify_logo_horizontal_black.jpg%26ehk%3dz9ClqCYg7MldmX8d1MAegCslS%252bAN0VU4sUH51qVgbjM%253d%26risl%3d%26pid%3dImgRaw&exph=1428&expw=3159&q=spotify&simid=608036742061648572&ck=24E3DADD5F5E8311DCABF0D95F8C58F0&selectedIndex=12&FORM=IRPRST&ajaxhist=0" width="40%" /> 

"""

app = FastAPI(
    title='Spotfy Song redictor API',
    description=description, 
    docs_url='/'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*']
)

classifier = load('app/classifier.joblib')


class Penguin(BaseModel):
    """Parse & validate penguin measurements"""
    bill_length_mm: confloat(gt=32, lt=60)
    bill_depth_mm: confloat(gt=13, lt=22)

    def to_df(self):
        """Convert to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])


@app.post('/predict')
def predict_species(penguin: Penguin):
    """Predict penguin species from bill length & depth
    
    Parameters
    ----------
    bill_length_mm : float, greater than 32, less than 60  
    bill_depth_mm : float, greater than 13, less than 22  

    Returns
    -------
    str "Adelie", "Chinstrap", or "Gentoo"  
    """
    species = classifier.predict(penguin.to_df())
    return species[0]


@app.get('/random')
def random_penguin():
    """Return a random penguin species"""
    species = random.choice(['Adelie', 'Chinstrap', 'Gentoo'])
    return species
