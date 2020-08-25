# House Price prediction using machine learning

## Introduction
The project consists of a dataset of 1460 unique houses and their 81 features such as their area, porch size, sunlight exposure, whether or not there is a bathtub and fireplace and much more which will be analysed to predict selling price of another 1459 houses.
Initially EDA has been performed to check relation of sale price with all the features and then a clean up the data to fill up the null values and then different algorithms have been used to find their accuracy in prediction.
The Dataset used is the Ames Housing dataset from Kaggle.


#### Importing Libraries


```python


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn= ignore_warn

from scipy import stats
from scipy.stats import norm, skew #for some statistics
```

### Importing the dataset


```python
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
print('Shape of train dataset: {}'.format(train.shape))
print('Shape of test dataset: {}'.format(test.shape))
```

    Shape of train dataset: (1460, 81)
    Shape of test dataset: (1459, 80)
    


```python
alldata=pd.concat((train,test), sort=False)
```


```python
alldata.shape
```




    (2919, 81)



### Exploratory Data Analysis


```python
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 100)

```


```python
alldata
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>856.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>284.0</td>
      <td>1262.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>920.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>756.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1145.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1936</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>4</td>
      <td>7</td>
      <td>1970</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1894</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>4</td>
      <td>5</td>
      <td>1970</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Rec</td>
      <td>252.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>294.0</td>
      <td>546.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>546</td>
      <td>546</td>
      <td>0</td>
      <td>1092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>CarPort</td>
      <td>1970.0</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>20</td>
      <td>RL</td>
      <td>160.0</td>
      <td>20000</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1960</td>
      <td>1996</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>ALQ</td>
      <td>1224.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1224</td>
      <td>0</td>
      <td>0</td>
      <td>1224</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Detchd</td>
      <td>1960.0</td>
      <td>Unf</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>474</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SFoyer</td>
      <td>5</td>
      <td>5</td>
      <td>1992</td>
      <td>1992</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>337.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>575.0</td>
      <td>912.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>970</td>
      <td>0</td>
      <td>0</td>
      <td>970</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>80</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1993</td>
      <td>1994</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>94.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>LwQ</td>
      <td>758.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>238.0</td>
      <td>996.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>996</td>
      <td>1004</td>
      <td>0</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1993.0</td>
      <td>Fin</td>
      <td>3.0</td>
      <td>650.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>190</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2919 rows × 81 columns</p>
</div>



#### Percentage of null values in the given data


```python
null_percent=pd.DataFrame(((alldata.isnull().sum()/alldata.shape[0]) * 100), columns=['Percent'])
null_percent.sort_values(by='Percent', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>99.657417</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>96.402878</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>93.216855</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>80.438506</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>49.982871</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>48.646797</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>16.649538</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>5.378554</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>2.809181</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>2.809181</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>2.774923</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>2.740665</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>2.706406</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.822199</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.787941</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>0.137033</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>SaleCondition</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PavedDrive</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Street</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotShape</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandContour</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotConfig</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition1</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition2</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BldgType</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HouseStyle</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofStyle</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofMatl</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ExterQual</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ExterCond</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Foundation</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Heating</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CentralAir</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HeatingQC</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Dropping columns with more than 50% missing data


```python
alldata.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)
```


```python
alldata.shape
```




    (2919, 77)




```python
null_percent=pd.DataFrame(((alldata.isnull().sum()/alldata.shape[0]) * 100), columns=['Percent'])
null_percent.sort_values(by='Percent', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SalePrice</th>
      <td>49.982871</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>48.646797</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>16.649538</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>5.378554</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>2.809181</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>2.809181</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>2.774923</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>2.740665</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>2.706406</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.822199</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.787941</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>0.137033</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>PavedDrive</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>SaleCondition</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Street</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotShape</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandContour</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotConfig</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition1</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition2</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BldgType</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HouseStyle</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofStyle</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofMatl</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ExterQual</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ExterCond</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Foundation</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HeatingQC</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CentralAir</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Heating</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Features with high correlation with the Sale Price


```python
high_corr=train.corr()
high_corr_features=high_corr.index[abs(high_corr['SalePrice'])>0.5]
high_corr_features
```




    Index(['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
           'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
           'SalePrice'],
          dtype='object')



####  Heatmap of high correlation features 


```python
local_corr=train[high_corr_features].corr()
plt.figure(figsize=(10,10))
sns.heatmap(local_corr, cmap='coolwarm', linewidth=2, annot=True)
```




    <AxesSubplot:>




![png](output_23_1.png)


#### Observing relationship of high correlation features using regression plot


```python
plt.figure(figsize=(16,9))
for i in range(len(high_corr_features)):
    plt.subplot(3,4,i+1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.regplot(x=high_corr_features[i], y='SalePrice', data=train)
```


![png](output_25_0.png)



```python
alldata.shape
```




    (2919, 77)



## Handling the missing data


```python
fix_columns=alldata.columns[alldata.isnull().any()]
fix_columns
```




    Index(['MSZoning', 'LotFrontage', 'Utilities', 'Exterior1st', 'Exterior2nd',
           'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
           'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
           'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
           'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
           'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
           'SaleType', 'SalePrice'],
          dtype='object')




```python
null_percent=pd.DataFrame(((alldata.isnull().sum()/alldata.shape[0]) * 100), columns=['Percent'])
null_percent.sort_values(by='Percent', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SalePrice</th>
      <td>49.982871</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>48.646797</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>16.649538</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>5.447071</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>5.378554</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>2.809181</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>2.809181</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>2.774923</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>2.740665</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>2.706406</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>0.822199</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>0.787941</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>0.137033</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>0.068517</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>0.034258</td>
    </tr>
    <tr>
      <th>PavedDrive</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>SaleCondition</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Street</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotShape</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandContour</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LotConfig</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition1</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Condition2</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>BldgType</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HouseStyle</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofStyle</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RoofMatl</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ExterQual</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ExterCond</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Foundation</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>HeatingQC</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CentralAir</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Heating</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Dividing the data into numerical and categorical features


```python
obj_col=alldata[fix_columns].columns[alldata[fix_columns].dtypes=='object']
obj_col_all=alldata.columns[alldata.dtypes=='object']

num_col=alldata[fix_columns].columns[alldata[fix_columns].dtypes!='object']
num_col_all=alldata.columns[alldata.dtypes!='object']

print('categorical columns with missing data: {} \n numerical columns with missing data: {}'.format(obj_col,num_col))
```

    categorical columns with missing data: Index(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
           'GarageFinish', 'GarageQual', 'GarageCond', 'SaleType'],
          dtype='object') 
     numerical columns with missing data: Index(['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
           'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
           'GarageCars', 'GarageArea', 'SalePrice'],
          dtype='object')
    

#### Cleaning categorical data


```python
null_percent2=pd.DataFrame(alldata[obj_col].isnull().sum(), columns=['missing'])
null_percent2.sort_values(by='missing', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FireplaceQu</th>
      <td>1420</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>159</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>157</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>82</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>82</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>81</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>80</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>79</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>24</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>2</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
NA = ['FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1'] 
alldata[NA] = alldata[NA].fillna('NA')
```


```python
lowNA=(alldata[obj_col].isnull().sum()>0).index.tolist()
for column in lowNA:
    mode=alldata[column].mode()[0]
    alldata[column].fillna(mode,inplace=True)
```


```python
alldata[obj_col].isnull().sum()
```




    MSZoning        0
    Utilities       0
    Exterior1st     0
    Exterior2nd     0
    MasVnrType      0
    BsmtQual        0
    BsmtCond        0
    BsmtExposure    0
    BsmtFinType1    0
    BsmtFinType2    0
    Electrical      0
    KitchenQual     0
    Functional      0
    FireplaceQu     0
    GarageType      0
    GarageFinish    0
    GarageQual      0
    GarageCond      0
    SaleType        0
    dtype: int64



#### Cleaning numerical data


```python
alldata[num_col].isnull().sum()
```




    LotFrontage      486
    MasVnrArea        23
    BsmtFinSF1         1
    BsmtFinSF2         1
    BsmtUnfSF          1
    TotalBsmtSF        1
    BsmtFullBath       2
    BsmtHalfBath       2
    GarageYrBlt      159
    GarageCars         1
    GarageArea         1
    SalePrice       1459
    dtype: int64




```python
alldata[num_col]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>196.0</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>856.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2003.0</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>208500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>0.0</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>284.0</td>
      <td>1262.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1976.0</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>181500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68.0</td>
      <td>162.0</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>920.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2001.0</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>223500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>756.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1998.0</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84.0</td>
      <td>350.0</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1145.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>250000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>21.0</td>
      <td>0.0</td>
      <td>252.0</td>
      <td>0.0</td>
      <td>294.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>160.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1224.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1960.0</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>62.0</td>
      <td>0.0</td>
      <td>337.0</td>
      <td>0.0</td>
      <td>575.0</td>
      <td>912.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>74.0</td>
      <td>94.0</td>
      <td>758.0</td>
      <td>0.0</td>
      <td>238.0</td>
      <td>996.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1993.0</td>
      <td>3.0</td>
      <td>650.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2919 rows × 12 columns</p>
</div>




```python
alldata['LotFrontage']=alldata['LotFrontage'].fillna(alldata['LotFrontage'].median())
```


```python
alldata[num_col].isnull().sum()
```




    LotFrontage        0
    MasVnrArea        23
    BsmtFinSF1         1
    BsmtFinSF2         1
    BsmtUnfSF          1
    TotalBsmtSF        1
    BsmtFullBath       2
    BsmtHalfBath       2
    GarageYrBlt      159
    GarageCars         1
    GarageArea         1
    SalePrice       1459
    dtype: int64




```python
plt.figure(figsize=(10,5))
sns.relplot(x='GarageYrBlt', y='YearBuilt', data=alldata)
sns.relplot(x='GarageYrBlt', y='YrSold', data=alldata)
```




    <seaborn.axisgrid.FacetGrid at 0x1d9d51e1580>




    <Figure size 720x360 with 0 Axes>



![png](output_42_2.png)



![png](output_42_3.png)



```python
GarYr=(alldata['YrSold']-alldata['YearBuilt']).median()
```


```python
alldata['GarageYrBlt']=alldata['GarageYrBlt'].fillna(alldata['YrSold']-35)
```


```python
alldata[alldata['MasVnrArea'].isnull()][['MasVnrArea','MasVnrType']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MasVnrArea</th>
      <th>MasVnrType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>234</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>529</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>650</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>936</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>973</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>977</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>231</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>246</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>422</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>532</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>544</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>581</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>851</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>865</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>880</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>889</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>908</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1402</th>
      <td>NaN</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
left=['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
last=alldata[left]
last.fillna(0,inplace=True)
alldata.update(last)
```


```python
alldata[left].isnull().sum()
```




    MasVnrArea      0
    BsmtFinSF1      0
    BsmtFinSF2      0
    BsmtUnfSF       0
    TotalBsmtSF     0
    BsmtFullBath    0
    BsmtHalfBath    0
    GarageCars      0
    GarageArea      0
    dtype: int64



#### Variance of the given data


```python
for col in alldata.columns[alldata.dtypes!='object']:
    print('{}    {:.2f}'.format(col,alldata[col].var()))
```

    Id    710290.00
    MSSubClass    1807.75
    LotFrontage    454.45
    LotArea    62204711.57
    OverallQual    1.99
    OverallCond    1.24
    YearBuilt    917.57
    YearRemodAdd    436.57
    MasVnrArea    31988.96
    BsmtFinSF1    207576.84
    BsmtFinSF2    28621.57
    BsmtUnfSF    193240.15
    TotalBsmtSF    194587.29
    1stFlrSF    153948.00
    2ndFlrSF    183784.94
    LowQualFinSF    2152.67
    GrLivArea    256087.66
    BsmtFullBath    0.28
    BsmtHalfBath    0.06
    FullBath    0.31
    HalfBath    0.25
    BedroomAbvGr    0.68
    KitchenAbvGr    0.05
    TotRmsAbvGrd    2.46
    Fireplaces    0.42
    GarageYrBlt    619.93
    GarageCars    0.58
    GarageArea    46455.63
    WoodDeckSF    16008.98
    OpenPorchSF    4566.45
    EnclosedPorch    4127.32
    3SsnPorch    634.44
    ScreenPorch    3156.68
    PoolArea    1271.92
    MiscVal    321945.27
    MoSold    7.37
    YrSold    1.73
    SalePrice    6311111264.30
    

#### Visualizing numerical data so as to drop features where one category is dominating others


```python
plt.figure(figsize=(40,140))
for i,num in zip(obj_col_all, range(len(obj_col_all))):
    plt.subplot(13,3,num+1)
    plt.bar(alldata[i].value_counts().index,alldata[i].value_counts())
    plt.title(i)
```


![png](output_51_0.png)



```python
alldata.drop(['Street','Utilities','Condition2','RoofMatl','Heating','PoolArea'], axis=1,inplace=True) #Dropping non varying parameters
```

#### Observing unique values of the categorical data


```python
for i in alldata[alldata.columns[alldata.dtypes=='object']]:
    print(i,alldata[alldata.columns[alldata.dtypes=='object']][i].unique(),'\n')
```

    MSZoning ['RL' 'RM' 'C (all)' 'FV' 'RH'] 
    
    LotShape ['Reg' 'IR1' 'IR2' 'IR3'] 
    
    LandContour ['Lvl' 'Bnk' 'Low' 'HLS'] 
    
    LotConfig ['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3'] 
    
    LandSlope ['Gtl' 'Mod' 'Sev'] 
    
    Neighborhood ['CollgCr' 'Veenker' 'Crawfor' 'NoRidge' 'Mitchel' 'Somerst' 'NWAmes'
     'OldTown' 'BrkSide' 'Sawyer' 'NridgHt' 'NAmes' 'SawyerW' 'IDOTRR'
     'MeadowV' 'Edwards' 'Timber' 'Gilbert' 'StoneBr' 'ClearCr' 'NPkVill'
     'Blmngtn' 'BrDale' 'SWISU' 'Blueste'] 
    
    Condition1 ['Norm' 'Feedr' 'PosN' 'Artery' 'RRAe' 'RRNn' 'RRAn' 'PosA' 'RRNe'] 
    
    BldgType ['1Fam' '2fmCon' 'Duplex' 'TwnhsE' 'Twnhs'] 
    
    HouseStyle ['2Story' '1Story' '1.5Fin' '1.5Unf' 'SFoyer' 'SLvl' '2.5Unf' '2.5Fin'] 
    
    RoofStyle ['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed'] 
    
    Exterior1st ['VinylSd' 'MetalSd' 'Wd Sdng' 'HdBoard' 'BrkFace' 'WdShing' 'CemntBd'
     'Plywood' 'AsbShng' 'Stucco' 'BrkComm' 'AsphShn' 'Stone' 'ImStucc'
     'CBlock'] 
    
    Exterior2nd ['VinylSd' 'MetalSd' 'Wd Shng' 'HdBoard' 'Plywood' 'Wd Sdng' 'CmentBd'
     'BrkFace' 'Stucco' 'AsbShng' 'Brk Cmn' 'ImStucc' 'AsphShn' 'Stone'
     'Other' 'CBlock'] 
    
    MasVnrType ['BrkFace' 'None' 'Stone' 'BrkCmn'] 
    
    ExterQual ['Gd' 'TA' 'Ex' 'Fa'] 
    
    ExterCond ['TA' 'Gd' 'Fa' 'Po' 'Ex'] 
    
    Foundation ['PConc' 'CBlock' 'BrkTil' 'Wood' 'Slab' 'Stone'] 
    
    BsmtQual ['Gd' 'TA' 'Ex' 'NA' 'Fa'] 
    
    BsmtCond ['TA' 'Gd' 'NA' 'Fa' 'Po'] 
    
    BsmtExposure ['No' 'Gd' 'Mn' 'Av' 'NA'] 
    
    BsmtFinType1 ['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' 'NA' 'LwQ'] 
    
    BsmtFinType2 ['Unf' 'BLQ' 'NA' 'ALQ' 'Rec' 'LwQ' 'GLQ'] 
    
    HeatingQC ['Ex' 'Gd' 'TA' 'Fa' 'Po'] 
    
    CentralAir ['Y' 'N'] 
    
    Electrical ['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix'] 
    
    KitchenQual ['Gd' 'TA' 'Ex' 'Fa'] 
    
    Functional ['Typ' 'Min1' 'Maj1' 'Min2' 'Mod' 'Maj2' 'Sev'] 
    
    FireplaceQu ['NA' 'TA' 'Gd' 'Fa' 'Ex' 'Po'] 
    
    GarageType ['Attchd' 'Detchd' 'BuiltIn' 'CarPort' 'NA' 'Basment' '2Types'] 
    
    GarageFinish ['RFn' 'Unf' 'Fin' 'NA'] 
    
    GarageQual ['TA' 'Fa' 'Gd' 'NA' 'Ex' 'Po'] 
    
    GarageCond ['TA' 'Fa' 'NA' 'Gd' 'Po' 'Ex'] 
    
    PavedDrive ['Y' 'N' 'P'] 
    
    SaleType ['WD' 'New' 'COD' 'ConLD' 'ConLI' 'CWD' 'ConLw' 'Con' 'Oth'] 
    
    SaleCondition ['Normal' 'Abnorml' 'Partial' 'AdjLand' 'Alloca' 'Family'] 
    
    

#### Features with similar type of data


```python
change=['ExterQual','BsmtExposure','BsmtFinType1','BsmtFinType2','ExterCond','HeatingQC','CentralAir','KitchenQual','FireplaceQu','GarageCond', 'GarageQual','BsmtQual','BsmtCond','LotShape']
for i in change:
    print(i,alldata[i].unique(),'\n')
len(change)
```

    ExterQual ['Gd' 'TA' 'Ex' 'Fa'] 
    
    BsmtExposure ['No' 'Gd' 'Mn' 'Av' 'NA'] 
    
    BsmtFinType1 ['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' 'NA' 'LwQ'] 
    
    BsmtFinType2 ['Unf' 'BLQ' 'NA' 'ALQ' 'Rec' 'LwQ' 'GLQ'] 
    
    ExterCond ['TA' 'Gd' 'Fa' 'Po' 'Ex'] 
    
    HeatingQC ['Ex' 'Gd' 'TA' 'Fa' 'Po'] 
    
    CentralAir ['Y' 'N'] 
    
    KitchenQual ['Gd' 'TA' 'Ex' 'Fa'] 
    
    FireplaceQu ['NA' 'TA' 'Gd' 'Fa' 'Ex' 'Po'] 
    
    GarageCond ['TA' 'Fa' 'NA' 'Gd' 'Po' 'Ex'] 
    
    GarageQual ['TA' 'Fa' 'Gd' 'NA' 'Ex' 'Po'] 
    
    BsmtQual ['Gd' 'TA' 'Ex' 'NA' 'Fa'] 
    
    BsmtCond ['TA' 'Gd' 'NA' 'Fa' 'Po'] 
    
    LotShape ['Reg' 'IR1' 'IR2' 'IR3'] 
    
    




    14



#### Assigning values as per their meaninng
Good given 5 and Poor given a score of 1 so that the machine learning model can understand the rating better and provide better prediction


```python
bin_map = { 'Gd':4, 'TA':3, 'Ex':5, 'Fa':2, 'Po':1, 'Y':1, 'N':0, 'GLQ':6, 'ALQ':5, 'Unf':1, 'Rec':3, 'BLQ':4, 'NA':0, 'LwQ':2,
           'No':1, 'Gd':3, 'Mn':2, 'Av':3, 'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0
}
for i in alldata[change]:
    alldata[i]=alldata[i].map(bin_map)
bin2={'N':0, 'P':1, 'Y':2}
alldata['PavedDrive'] =alldata['PavedDrive'].map(bin2)

```


```python
for i in change:
    print(i,alldata[i].unique(),'\n')
len(change)
```

    ExterQual [3 5 2] 
    
    BsmtExposure [1 3 2 0] 
    
    BsmtFinType1 [6 5 1 3 4 0 2] 
    
    BsmtFinType2 [1 4 0 5 3 2 6] 
    
    ExterCond [3 2 1 5] 
    
    HeatingQC [5 3 2 1] 
    
    CentralAir [1 0] 
    
    KitchenQual [3 5 2] 
    
    FireplaceQu [0 3 2 5 1] 
    
    GarageCond [3 2 0 1 5] 
    
    GarageQual [3 2 0 5 1] 
    
    BsmtQual [3 5 0 2] 
    
    BsmtCond [3 0 2 1] 
    
    LotShape [3 2 1 0] 
    
    




    14



#### Checking the heatmap of the features just transformed into numerical ranks


```python
change2=change
change2.append('SalePrice')
change2.append('PavedDrive')

change_corr=alldata[change2].corr()
plt.figure(figsize=(10,10))
sns.heatmap(change_corr, cmap='coolwarm', linewidth=2, annot=True)
```




    <AxesSubplot:>




![png](output_61_1.png)


#### Creating Some new features


```python
alldata['House_age']=alldata['YrSold']-alldata['YearBuilt']
alldata['House_age'].describe()
```




    count    2919.000000
    mean       36.479959
    std        30.336182
    min        -1.000000
    25%         7.000000
    50%        35.000000
    75%        54.500000
    max       136.000000
    Name: House_age, dtype: float64



#### Checking if the new feature created does not contain any absurd value such as a negative age


```python
alldata[alldata['House_age'] < 0] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>House_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1089</th>
      <td>2550</td>
      <td>20</td>
      <td>RL</td>
      <td>128.0</td>
      <td>39290</td>
      <td>2</td>
      <td>Bnk</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>10</td>
      <td>5</td>
      <td>2008</td>
      <td>2009</td>
      <td>Hip</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>Stone</td>
      <td>1224.0</td>
      <td>5</td>
      <td>3</td>
      <td>PConc</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>4010.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1085.0</td>
      <td>5095.0</td>
      <td>5</td>
      <td>1</td>
      <td>SBrkr</td>
      <td>5095</td>
      <td>0</td>
      <td>0</td>
      <td>5095</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>15</td>
      <td>Typ</td>
      <td>2</td>
      <td>3</td>
      <td>Attchd</td>
      <td>2008.0</td>
      <td>Fin</td>
      <td>3.0</td>
      <td>1154.0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>546</td>
      <td>484</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17000</td>
      <td>10</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
alldata.iloc[1089]['YrSold']=2010
```


```python
alldata['House_age']=alldata['YrSold']-alldata['YearBuilt']

```


```python
alldata[alldata['SalePrice'].isnull()].shape
```




    (1459, 72)




```python
alldata[alldata['SalePrice'].notnull()].shape
```




    (1460, 72)




```python
test=alldata[alldata['SalePrice'].isnull()]
```


```python
train=alldata[alldata['SalePrice'].notnull()]
```


```python
train.drop([1379,1298,523],inplace=True)
```


```python
alldata2=pd.concat((train,test), sort=False)
```

#### Heatmap of complete data


```python
corrmat=alldata2.corr()
f, ax = plt.subplots(figsize=(16,13))
sns.heatmap(corrmat,linewidth=0.2,vmax=0.8,cmap='coolwarm')
```




    <AxesSubplot:>




![png](output_75_1.png)



```python
most_corr_feat=corrmat.index[abs(corrmat['SalePrice'])>0.5]
high_corr=alldata[most_corr_feat].corr()
```

#### Heatmap of features with correlation greater than 0.5


```python
plt.figure(figsize=(10,8))
sns.heatmap(high_corr,linewidth=0.2,vmax=0.8,annot=True,cmap='coolwarm')
```




    <AxesSubplot:>




![png](output_78_1.png)



```python
#alldata2.drop(['GarageArea','TotRmsAbvGrd','TotalBsmtSF'],axis=1,inplace=True) #Clearing for multicollinearity
```


```python
vhigh_corr_feat=corrmat.index[abs(corrmat['SalePrice'])>0.55].tolist()
vhigh_corr_feat
```




    ['OverallQual',
     'TotalBsmtSF',
     '1stFlrSF',
     'GrLivArea',
     'FullBath',
     'GarageCars',
     'GarageArea',
     'SalePrice']



#### Pairplot showing distribution of the data and trend follwed by the data with some of the features


```python
sns.set()
sns.pairplot(alldata2[vhigh_corr_feat], size=2.5)
plt.show()
```


![png](output_82_0.png)


#### Observing the distribution of the Sale Price data and calculating its skewness


```python
plt.figure(figsize=(10,8))
bar=sns.distplot(alldata2['SalePrice'], fit=norm)
mu,sigma=norm.fit(train['SalePrice'])
bar.legend(['Skewness: {:.2f}\nNormal Distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(alldata2['SalePrice'].skew(),mu,sigma)], loc='best');

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```


![png](output_84_0.png)



![png](output_84_1.png)



```python
alldata2.drop('House_age', axis=1,inplace=True)
```


```python
skewed=[]
for i in alldata2.columns[alldata2.dtypes!='object']:
    sk=alldata2[i].skew()
    if abs(sk) > 0.5:
        skewed.append(i)

```

#### Features with skewness greater than 0.5


```python
skewed
```




    ['MSSubClass',
     'LotFrontage',
     'LotArea',
     'LotShape',
     'OverallCond',
     'YearBuilt',
     'MasVnrArea',
     'ExterQual',
     'BsmtCond',
     'BsmtExposure',
     'BsmtFinSF1',
     'BsmtFinType2',
     'BsmtFinSF2',
     'BsmtUnfSF',
     'TotalBsmtSF',
     'CentralAir',
     '1stFlrSF',
     '2ndFlrSF',
     'LowQualFinSF',
     'GrLivArea',
     'BsmtFullBath',
     'BsmtHalfBath',
     'HalfBath',
     'KitchenAbvGr',
     'KitchenQual',
     'TotRmsAbvGrd',
     'Fireplaces',
     'GarageQual',
     'GarageCond',
     'PavedDrive',
     'WoodDeckSF',
     'OpenPorchSF',
     'EnclosedPorch',
     '3SsnPorch',
     'ScreenPorch',
     'MiscVal',
     'SalePrice']




```python
for i in skewed:
    alldata2[i]=np.log(alldata2[i]+1)
```


```python
vhigh_corr_feat.remove('SalePrice')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-100-c8032b68756a> in <module>
    ----> 1 vhigh_corr_feat.remove('SalePrice')
    

    ValueError: list.remove(x): x not in list



```python
for i in vhigh_corr_feat:
    plt.figure()
    bar=sns.distplot(alldata2[i], fit=norm)
    mu,sigma=norm.fit(alldata2[i])
    bar.legend(['Skewness: {:.2f}\nNormal Distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(alldata2[i].skew(),mu,sigma)], loc='best');
    plt.title(i)
    
```


![png](output_91_0.png)



![png](output_91_1.png)



![png](output_91_2.png)



![png](output_91_3.png)



![png](output_91_4.png)



![png](output_91_5.png)



![png](output_91_6.png)



```python
object_rem_col=alldata2.columns[alldata2.dtypes=='object'].tolist()
num_rem_col=alldata2.columns[alldata2.dtypes!='object'].tolist()
```


```python
num_df=alldata2[num_rem_col]
object_df=alldata2[object_rem_col]
object_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>RoofStyle</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>Foundation</th>
      <th>Electrical</th>
      <th>Functional</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>CBlock</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>BrkTil</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>RM</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>CBlock</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>NA</td>
      <td>NA</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>RM</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>CBlock</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>CarPort</td>
      <td>Unf</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>CBlock</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SFoyer</td>
      <td>Gable</td>
      <td>HdBoard</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>PConc</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>NA</td>
      <td>NA</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>RL</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>BrkFace</td>
      <td>PConc</td>
      <td>SBrkr</td>
      <td>Typ</td>
      <td>Attchd</td>
      <td>Fin</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>2916 rows × 19 columns</p>
</div>




```python
num_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>LotShape</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4.110874</td>
      <td>4.189655</td>
      <td>9.042040</td>
      <td>1.386294</td>
      <td>7</td>
      <td>1.791759</td>
      <td>7.602900</td>
      <td>2003</td>
      <td>5.283204</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>6</td>
      <td>6.561031</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>5.017280</td>
      <td>6.753438</td>
      <td>5</td>
      <td>0.693147</td>
      <td>6.753438</td>
      <td>6.751101</td>
      <td>0.0</td>
      <td>7.444833</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.693147</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>2.197225</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2003.0</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>4.127134</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2</td>
      <td>2008</td>
      <td>12.247699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3.044522</td>
      <td>4.394449</td>
      <td>9.169623</td>
      <td>1.386294</td>
      <td>6</td>
      <td>2.197225</td>
      <td>7.589336</td>
      <td>1976</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>5</td>
      <td>6.886532</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>5.652489</td>
      <td>7.141245</td>
      <td>5</td>
      <td>0.693147</td>
      <td>7.141245</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.141245</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>2</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>1.945910</td>
      <td>0.693147</td>
      <td>3</td>
      <td>1976.0</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>5.700444</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>5</td>
      <td>2007</td>
      <td>12.109016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.110874</td>
      <td>4.234107</td>
      <td>9.328212</td>
      <td>1.098612</td>
      <td>7</td>
      <td>1.791759</td>
      <td>7.601902</td>
      <td>2002</td>
      <td>5.093750</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>6</td>
      <td>6.188264</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>6.075346</td>
      <td>6.825460</td>
      <td>5</td>
      <td>0.693147</td>
      <td>6.825460</td>
      <td>6.765039</td>
      <td>0.0</td>
      <td>7.488294</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.693147</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>1.945910</td>
      <td>0.693147</td>
      <td>3</td>
      <td>2001.0</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>3.761200</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>9</td>
      <td>2008</td>
      <td>12.317171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.262680</td>
      <td>4.110874</td>
      <td>9.164401</td>
      <td>1.098612</td>
      <td>7</td>
      <td>1.791759</td>
      <td>7.557995</td>
      <td>1970</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>5</td>
      <td>5.379897</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>6.293419</td>
      <td>6.629363</td>
      <td>3</td>
      <td>0.693147</td>
      <td>6.869014</td>
      <td>6.629363</td>
      <td>0.0</td>
      <td>7.448916</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>2.079442</td>
      <td>0.693147</td>
      <td>3</td>
      <td>1998.0</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>3.583519</td>
      <td>5.609472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2</td>
      <td>2006</td>
      <td>11.849405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4.110874</td>
      <td>4.442651</td>
      <td>9.565284</td>
      <td>1.098612</td>
      <td>8</td>
      <td>1.791759</td>
      <td>7.601402</td>
      <td>2000</td>
      <td>5.860786</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>6</td>
      <td>6.486161</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>6.196444</td>
      <td>7.044033</td>
      <td>5</td>
      <td>0.693147</td>
      <td>7.044033</td>
      <td>6.960348</td>
      <td>0.0</td>
      <td>7.695758</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.693147</td>
      <td>4</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>2.302585</td>
      <td>0.693147</td>
      <td>3</td>
      <td>2000.0</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>5.262690</td>
      <td>4.442651</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>12</td>
      <td>2008</td>
      <td>12.429220</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>5.081404</td>
      <td>3.091042</td>
      <td>7.568896</td>
      <td>1.386294</td>
      <td>4</td>
      <td>2.079442</td>
      <td>7.586296</td>
      <td>1970</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>6.304449</td>
      <td>6.304449</td>
      <td>3</td>
      <td>0.693147</td>
      <td>6.304449</td>
      <td>6.304449</td>
      <td>0.0</td>
      <td>6.996681</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.693147</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>1.791759</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1971.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>6</td>
      <td>2006</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>5.081404</td>
      <td>3.091042</td>
      <td>7.546974</td>
      <td>1.386294</td>
      <td>4</td>
      <td>1.791759</td>
      <td>7.586296</td>
      <td>1970</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>3</td>
      <td>5.533389</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>5.686975</td>
      <td>6.304449</td>
      <td>3</td>
      <td>0.693147</td>
      <td>6.304449</td>
      <td>6.304449</td>
      <td>0.0</td>
      <td>6.996681</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.693147</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>1.945910</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>286.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>3.218876</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>4</td>
      <td>2006</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>3.044522</td>
      <td>5.081404</td>
      <td>9.903538</td>
      <td>1.386294</td>
      <td>5</td>
      <td>2.079442</td>
      <td>7.581210</td>
      <td>1996</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>5</td>
      <td>7.110696</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7.110696</td>
      <td>5</td>
      <td>0.693147</td>
      <td>7.110696</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.110696</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>4</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>2.079442</td>
      <td>0.693147</td>
      <td>3</td>
      <td>1960.0</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>6.163315</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>9</td>
      <td>2006</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>4.454347</td>
      <td>4.143135</td>
      <td>9.253591</td>
      <td>1.386294</td>
      <td>5</td>
      <td>1.791759</td>
      <td>7.597396</td>
      <td>1992</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>6</td>
      <td>5.823046</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>6.356108</td>
      <td>6.816736</td>
      <td>3</td>
      <td>0.693147</td>
      <td>6.878326</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>6.878326</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>1</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>1.945910</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1971.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>4.394449</td>
      <td>3.496508</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.552508</td>
      <td>7</td>
      <td>2006</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>4.110874</td>
      <td>4.317488</td>
      <td>9.172431</td>
      <td>1.386294</td>
      <td>7</td>
      <td>1.791759</td>
      <td>7.597898</td>
      <td>1994</td>
      <td>4.553877</td>
      <td>1.386294</td>
      <td>3</td>
      <td>3</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>2</td>
      <td>6.632002</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>5.476464</td>
      <td>6.904751</td>
      <td>5</td>
      <td>0.693147</td>
      <td>6.904751</td>
      <td>6.912743</td>
      <td>0.0</td>
      <td>7.601402</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2</td>
      <td>0.693147</td>
      <td>3</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>2.302585</td>
      <td>0.693147</td>
      <td>3</td>
      <td>1993.0</td>
      <td>3.0</td>
      <td>650.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>5.252273</td>
      <td>3.891820</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>11</td>
      <td>2006</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2916 rows × 52 columns</p>
</div>



#### Applying get dummies to conver remaining object data to numerical



```python
object_df=pd.get_dummies(object_df)
object_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning_C (all)</th>
      <th>MSZoning_FV</th>
      <th>MSZoning_RH</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>LandContour_Bnk</th>
      <th>LandContour_HLS</th>
      <th>LandContour_Low</th>
      <th>LandContour_Lvl</th>
      <th>LotConfig_Corner</th>
      <th>LotConfig_CulDSac</th>
      <th>LotConfig_FR2</th>
      <th>LotConfig_FR3</th>
      <th>LotConfig_Inside</th>
      <th>LandSlope_Gtl</th>
      <th>LandSlope_Mod</th>
      <th>LandSlope_Sev</th>
      <th>Neighborhood_Blmngtn</th>
      <th>Neighborhood_Blueste</th>
      <th>Neighborhood_BrDale</th>
      <th>Neighborhood_BrkSide</th>
      <th>Neighborhood_ClearCr</th>
      <th>Neighborhood_CollgCr</th>
      <th>Neighborhood_Crawfor</th>
      <th>Neighborhood_Edwards</th>
      <th>Neighborhood_Gilbert</th>
      <th>Neighborhood_IDOTRR</th>
      <th>Neighborhood_MeadowV</th>
      <th>Neighborhood_Mitchel</th>
      <th>Neighborhood_NAmes</th>
      <th>Neighborhood_NPkVill</th>
      <th>Neighborhood_NWAmes</th>
      <th>Neighborhood_NoRidge</th>
      <th>Neighborhood_NridgHt</th>
      <th>Neighborhood_OldTown</th>
      <th>Neighborhood_SWISU</th>
      <th>Neighborhood_Sawyer</th>
      <th>Neighborhood_SawyerW</th>
      <th>Neighborhood_Somerst</th>
      <th>Neighborhood_StoneBr</th>
      <th>Neighborhood_Timber</th>
      <th>Neighborhood_Veenker</th>
      <th>Condition1_Artery</th>
      <th>Condition1_Feedr</th>
      <th>Condition1_Norm</th>
      <th>Condition1_PosA</th>
      <th>Condition1_PosN</th>
      <th>Condition1_RRAe</th>
      <th>Condition1_RRAn</th>
      <th>Condition1_RRNe</th>
      <th>Condition1_RRNn</th>
      <th>BldgType_1Fam</th>
      <th>BldgType_2fmCon</th>
      <th>BldgType_Duplex</th>
      <th>BldgType_Twnhs</th>
      <th>BldgType_TwnhsE</th>
      <th>HouseStyle_1.5Fin</th>
      <th>HouseStyle_1.5Unf</th>
      <th>HouseStyle_1Story</th>
      <th>HouseStyle_2.5Fin</th>
      <th>HouseStyle_2.5Unf</th>
      <th>HouseStyle_2Story</th>
      <th>HouseStyle_SFoyer</th>
      <th>HouseStyle_SLvl</th>
      <th>RoofStyle_Flat</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Gambrel</th>
      <th>RoofStyle_Hip</th>
      <th>RoofStyle_Mansard</th>
      <th>RoofStyle_Shed</th>
      <th>Exterior1st_AsbShng</th>
      <th>Exterior1st_AsphShn</th>
      <th>Exterior1st_BrkComm</th>
      <th>Exterior1st_BrkFace</th>
      <th>Exterior1st_CBlock</th>
      <th>Exterior1st_CemntBd</th>
      <th>Exterior1st_HdBoard</th>
      <th>Exterior1st_ImStucc</th>
      <th>Exterior1st_MetalSd</th>
      <th>Exterior1st_Plywood</th>
      <th>Exterior1st_Stone</th>
      <th>Exterior1st_Stucco</th>
      <th>Exterior1st_VinylSd</th>
      <th>Exterior1st_Wd Sdng</th>
      <th>Exterior1st_WdShing</th>
      <th>Exterior2nd_AsbShng</th>
      <th>Exterior2nd_AsphShn</th>
      <th>Exterior2nd_Brk Cmn</th>
      <th>Exterior2nd_BrkFace</th>
      <th>Exterior2nd_CBlock</th>
      <th>Exterior2nd_CmentBd</th>
      <th>Exterior2nd_HdBoard</th>
      <th>Exterior2nd_ImStucc</th>
      <th>Exterior2nd_MetalSd</th>
      <th>Exterior2nd_Other</th>
      <th>Exterior2nd_Plywood</th>
      <th>Exterior2nd_Stone</th>
      <th>Exterior2nd_Stucco</th>
      <th>Exterior2nd_VinylSd</th>
      <th>Exterior2nd_Wd Sdng</th>
      <th>Exterior2nd_Wd Shng</th>
      <th>MasVnrType_BrkCmn</th>
      <th>MasVnrType_BrkFace</th>
      <th>MasVnrType_None</th>
      <th>MasVnrType_Stone</th>
      <th>Foundation_BrkTil</th>
      <th>Foundation_CBlock</th>
      <th>Foundation_PConc</th>
      <th>Foundation_Slab</th>
      <th>Foundation_Stone</th>
      <th>Foundation_Wood</th>
      <th>Electrical_FuseA</th>
      <th>Electrical_FuseF</th>
      <th>Electrical_FuseP</th>
      <th>Electrical_Mix</th>
      <th>Electrical_SBrkr</th>
      <th>Functional_Maj1</th>
      <th>Functional_Maj2</th>
      <th>Functional_Min1</th>
      <th>Functional_Min2</th>
      <th>Functional_Mod</th>
      <th>Functional_Sev</th>
      <th>Functional_Typ</th>
      <th>GarageType_2Types</th>
      <th>GarageType_Attchd</th>
      <th>GarageType_Basment</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_CarPort</th>
      <th>GarageType_Detchd</th>
      <th>GarageType_NA</th>
      <th>GarageFinish_Fin</th>
      <th>GarageFinish_NA</th>
      <th>GarageFinish_RFn</th>
      <th>GarageFinish_Unf</th>
      <th>SaleType_COD</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2916 rows × 149 columns</p>
</div>




```python
finaldf=pd.concat([num_df,object_df],axis=1,sort=False)
finaldf1=finaldf.copy()
finaldf.drop(['Id','SalePrice'],axis=1,inplace=True)
SalePrice=finaldf1['SalePrice'][:1457]
test1=finaldf1[1457:]

```


```python
train=finaldf[:1457]
test=finaldf[1457:]

test.shape
```




    (1459, 199)



### Scaling the data


```python
from sklearn.preprocessing import RobustScaler
rscaler = RobustScaler()
rscaler.fit(finaldf)
finaldf = rscaler.transform(finaldf)
```


```python
finaldf.shape
```




    (2916, 199)




```python
X_train=finaldf[:1457]
X_test=finaldf[1457:]
y_train=SalePrice
```

### Calculating cross validation score


```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score
```

#### Applying Linear Regression 


```python
import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)
```




    [-1.380581673686194e+19]




```python
# Cross validation
cross_validation = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
print("Cross validation accuracy of LR model = ", cross_validation)
print("\nCross validation mean accuracy of LR model = ", cross_validation.mean())
```

    Cross validation accuracy of LR model =  [ 9.06162369e-01  9.39823896e-01 -4.68938536e+17  9.12424080e-01
     -4.97724596e+18  9.09665282e-01 -3.34095594e+16 -5.44931520e+17
     -6.49961912e+18 -1.44479832e+17]
    
    Cross validation mean accuracy of LR model =  -1.266862452527026e+18
    

#### Applying Ridge Regression


```python
rdg = linear_model.Ridge()
test_model(rdg)
```




    [0.9142252520961517]



#### Applying Lasso Regression


```python
lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)
```




    [0.9148962694339445]



#### Applying support vector machine


```python
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)
```




    [0.863976840825292]



#### Applying Decision Tree regressor


```python
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)
```




    [0.7265624152814425]



#### Applying Random Forest Regressor


```python
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)
```




    [0.8689815661500444]



#### Applying Bagging and Gradient Boost Regressor


```python
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
br_reg = BaggingRegressor(n_estimators=1000, random_state=51)
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, loss='ls', random_state=51)
```


```python
test_model(br_reg)

```




    [0.8688654990207884]




```python
test_model(gbr_reg)

```




    [0.9072802333137965]



Gradient boost regressor provided a score of 90% Hence we will predict the values using it.

#### Applying XGBoost


```python
import xgboost
#xgb_reg=xgboost.XGBRegressor()
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)
```

    [17:33:30] WARNING: D:\Build\xgboost\xgboost-1.1.0.git\python-package\build\temp.win-amd64-3.8\Release\xgboost\src\learner.cc:480: 
    Parameters: { bbooster } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:33:30] WARNING: D:\Build\xgboost\xgboost-1.1.0.git\python-package\build\temp.win-amd64-3.8\Release\xgboost\src\learner.cc:480: 
    Parameters: { bbooster } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:33:31] WARNING: D:\Build\xgboost\xgboost-1.1.0.git\python-package\build\temp.win-amd64-3.8\Release\xgboost\src\learner.cc:480: 
    Parameters: { bbooster } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    




    [0.8851231233151481]



### Training and testing the dataset on gradient boost regressor


```python
gbr_reg.fit(X_train,y_train)
y_pred=np.exp(gbr_reg.predict(X_test)).round(2)
submit_test=pd.concat([test1['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
```


```python
np.exp(gbr_reg.predict(X_test))
```




    array([121688.57275478, 157078.24174323, 187833.03855412, ...,
           143350.20146421, 117565.74688315, 234240.19289512])




```python
submit_test.to_csv('sample_submission1', index=False)
```


```python
submit_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>121688.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>157078.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>187833.04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>198800.19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>179660.02</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>77312.84</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>75398.44</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>143350.20</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>117565.75</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>234240.19</td>
    </tr>
  </tbody>
</table>
<p>1459 rows × 2 columns</p>
</div>


