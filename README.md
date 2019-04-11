
# Fama French 3-Factor Model
By Robert Yip  
Oct 2018  
Built with Python

In this project, I build a Fama French 3-factor model using two opposite portfolios from Morningstar. The first portfolio is based on an Aggressive strategy and the other a Conservative strategy. The results show the model efficacy based on the strength of the fit.

The project includes these steps:
1) Retrieval - Sourcing the raw data from CSV files (monthly snapshots of the portfolio). The CSVs are retrieved and exported from SQL. Multiple functions are defining to aid in the retrieval and table transformation.
2) Transformation - Setup the tables properly to calculate returns and categorize portfolio constituents to factors
3) Regression model

The Fama French 3-factor model has these attributes. The description explains the proxy that I used.  
**Market Premium** - Calculated from S&P/TSX Composite Index and 90-day Treasury Bills.  
**SMB** - Categorized each security as small or large market cap by using 30-70 percentiles of aggregate market cap in portfolio.  
**HML** - Used inverse of P/B as proxy to catergorize and calculate book to market value.  


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
pd.options.mode.chained_assignment = None  # default='warn'
```

    C:\Users\Ry\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    


```python
#Set years of data to look at: 2007-2018
start = 2007
end = 2018
```

## 1) Retrieving the data
```python
def dfAppend(strat, s, e):
    """
    Reads data and appends to a central df.
    df    = data frame to append to
    strat = Strategies label {A = Aggressive, C = Conservative}
    start = year to start
    end   = year to end, not inclusive of end date
    """
    df = pd.DataFrame()
    fileRange = np.arange(s, e + 1)     
    
    for i in fileRange:
        if i == s:
            dfAdd = pd.read_csv(strat + str(i) + ".csv", skiprows=range(1,4), sep=',', encoding='iso-8859-1')
        else:
            dfAdd = pd.read_csv(strat + str(i) + ".csv", skiprows=range(1,4), sep=',', encoding='iso-8859-1')
        dfAdd = dfAdd.iloc[:, : -1] #drops last extra column
        dfAdd['Year'] = i
        df = df.append(dfAdd, ignore_index = True)

    return df
```
##2) Transformation

```python
def dfResetIndex (df):
    """
    Resets index each time a df is made
    """
    
    return df.reset_index(drop = True)
```


```python
def calcReturn(df):
    """
    Gives return of individual security from portfolio
    """
   
    return round(df.Return.mean(), 4)
```


```python
def calcRf (df):
    """
    Returns Rf for FF
    """
    df = dfResetIndex (df)
    return round(df.TB90[1], 4)
```


```python
def fillMktPrem (df, s, e):
    """
    Fills in MktPrem to DF
    """
    dfNew = df
    dfNew['MKtReturn'] = ""
    dfNew['MktPrem'] = ""
    fileRange = np.arange(s, e)    
    for i in fileRange:
        dfNew['MKtReturn'].loc[dfNew['Year'] == i+1] = (dfNew['TRI'].loc[dfNew['Year'] == i+1].iloc[0] / dfNew['TRI'].loc[dfNew['Year'] == i].iloc[0] - 1)*100
        dfNew['MktPrem'].loc[dfNew['Year'] == i+1] = dfNew['MKtReturn'].loc[dfNew['Year'] == i+1].iloc[0] - dfNew['TB90'].loc[dfNew['Year'] == i+1].iloc[0]
    return dfNew
```


```python
def calcMktPrem (df):
    """
    Returns Mkt Premium for FF
    """
    df = dfResetIndex(df)
    return round(df.MktPrem[1], 4)
```


```python
def calcSMB(df):
    """
    Returns SMB for FF
    """
    #Define Quantile
    SQuantile = 0.3
    LQuantile = 0.7
    df["SMB"] = ""
    
    #Assigns stock size based on market cap
    df.SMB[df.MKTCAP <= df.MKTCAP.quantile(SQuantile)] = "SCap"
    df.SMB[(df.MKTCAP > df.MKTCAP.quantile(SQuantile)) & (df.MKTCAP < df.MKTCAP.quantile(LQuantile))] = "MCap"
    df.SMB[df.MKTCAP >= df.MKTCAP.quantile(LQuantile)] = "LCap"
    
    #Calculates average return of stocks in portfolio subset based on size
    SmallCapReturn = df.Return.loc[df["SMB"] == "SCap"].mean()
    LargeCapReturn = df.Return.loc[df["SMB"] == "LCap"].mean()
    
    #Returns SMB based on definition
    SMB = SmallCapReturn - LargeCapReturn
    return round(SMB, 4)

```


```python
def calcHML (df):
    """
    Returns HML for FF
    Uses inverse of P/B as proxy for Book/Mkt
    """
    #Define Quantile
    SQuantile = 0.3
    LQuantile = 0.7
    df["HML"] = ""
    df["BP"] = df.PB**(-1) #Create Book/MktValue Proxy
    
    #Assigns stock size based on market cap
    df.HML[df.BP <= df.BP.quantile(SQuantile)] = "SValue"
    df.HML[(df.BP > df.BP.quantile(SQuantile)) & (df.BP < df.BP.quantile(LQuantile))] = "MValue"
    df.HML[df.BP >= df.BP.quantile(LQuantile)] = "LValue"
    
    #Calculates average return of stocks in portfolio subset based on size
    SmallValueReturn = df.Return.loc[df["HML"] == "SValue"].mean()
    LargeValueReturn = df.Return.loc[df["HML"] == "LValue"].mean()
    
    #Returns SMB based on definition
    HML = SmallValueReturn - LargeValueReturn
    return round(HML, 4)
```


```python
def cleanColumns(df):
    """
    Cleans up unnecessary characters
    Cleans up columns, removing the extras
    """
    dfNew = df
   
    try:
        dfNew.columns = dfNew.columns.str.replace(' ','')
    except:
        pass
    
    try:
        dfNew.columns = dfNew.columns.str.replace('/','')
    except:
        pass
    
    dfNew = dfNew.rename(columns={"PCHG12M": "Return"})
    
    dfNew = dfNew[['Symbol',
                'Year',
                'Return',
                'TRI',
                'TB90',
                'MKTCAP',
                'PB'
                ]]
    
    return dfNew   
```


```python
###Set up Data Frame

#Create empty data frame for the strategies
dfA = pd.DataFrame()
dfC = pd.DataFrame()

#Append the list
dfA = dfAppend("A", start, end) #year 2007-2018
dfC = dfAppend("C", start, end) #year 2008-2018

```


```python
###Clean up Data Frame and preparing for FF model
#Remove space in columns
dfA = cleanColumns(dfA)
dfC = cleanColumns(dfC)

dfA
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Year</th>
      <th>Return</th>
      <th>TRI</th>
      <th>TB90</th>
      <th>MKTCAP</th>
      <th>PB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TCM</td>
      <td>2007</td>
      <td>394.1667</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>2009.5763</td>
      <td>4.8272</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BB</td>
      <td>2007</td>
      <td>196.5454</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>50392.2070</td>
      <td>17.3708</td>
    </tr>
    <tr>
      <th>2</th>
      <td>QUX</td>
      <td>2007</td>
      <td>52.4324</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>914.2384</td>
      <td>2.0376</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRE</td>
      <td>2007</td>
      <td>300.0000</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>2908.6021</td>
      <td>2.6323</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VT</td>
      <td>2007</td>
      <td>46.1039</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>2296.7551</td>
      <td>2.1184</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LNR</td>
      <td>2007</td>
      <td>89.3727</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1791.8121</td>
      <td>2.0731</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MDI</td>
      <td>2007</td>
      <td>102.1609</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1035.5814</td>
      <td>4.6532</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ET</td>
      <td>2007</td>
      <td>112.4800</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1918.1101</td>
      <td>17.0584</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SW</td>
      <td>2007</td>
      <td>89.2187</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>662.2232</td>
      <td>3.3381</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NTR</td>
      <td>2007</td>
      <td>158.3199</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>29561.9219</td>
      <td>6.0249</td>
    </tr>
    <tr>
      <th>10</th>
      <td>OIL1</td>
      <td>2007</td>
      <td>90.4130</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>2810.2747</td>
      <td>5.2046</td>
    </tr>
    <tr>
      <th>11</th>
      <td>VRS1</td>
      <td>2007</td>
      <td>325.7028</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>622.7500</td>
      <td>3.6446</td>
    </tr>
    <tr>
      <th>12</th>
      <td>AXP1</td>
      <td>2007</td>
      <td>31.8211</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1124.9443</td>
      <td>1.5986</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SVC</td>
      <td>2007</td>
      <td>n/a</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>771.8656</td>
      <td>5.4759</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ELR</td>
      <td>2007</td>
      <td>80.3150</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1528.5818</td>
      <td>1.8666</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AGU</td>
      <td>2007</td>
      <td>87.9626</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>6445.6694</td>
      <td>4.0877</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MG</td>
      <td>2007</td>
      <td>19.5411</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>10483.0146</td>
      <td>1.2566</td>
    </tr>
    <tr>
      <th>17</th>
      <td>AL</td>
      <td>2007</td>
      <td>109.7164</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>38855.3320</td>
      <td>3.0089</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NCX</td>
      <td>2007</td>
      <td>9.1915</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>3194.3237</td>
      <td>3.4441</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EQN</td>
      <td>2007</td>
      <td>120.0000</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1972.0448</td>
      <td>3.9819</td>
    </tr>
    <tr>
      <th>20</th>
      <td>IOL</td>
      <td>2007</td>
      <td>111.3636</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1113.1356</td>
      <td>10.1440</td>
    </tr>
    <tr>
      <th>21</th>
      <td>SCL</td>
      <td>2007</td>
      <td>85.8289</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>2491.2971</td>
      <td>4.3214</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SCC</td>
      <td>2007</td>
      <td>27.5294</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>2916.5293</td>
      <td>3.4336</td>
    </tr>
    <tr>
      <th>23</th>
      <td>OTEX</td>
      <td>2007</td>
      <td>45.7270</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1318.4620</td>
      <td>2.3879</td>
    </tr>
    <tr>
      <th>24</th>
      <td>EMP.A</td>
      <td>2007</td>
      <td>12.1356</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>1547.3501</td>
      <td>1.5353</td>
    </tr>
    <tr>
      <th>25</th>
      <td>SAP</td>
      <td>2007</td>
      <td>34.4371</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>5194.7700</td>
      <td>3.6509</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ACO.X</td>
      <td>2007</td>
      <td>35.3991</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>3264.8979</td>
      <td>2.1489</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AGF.B</td>
      <td>2007</td>
      <td>60.0913</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>3164.1650</td>
      <td>3.0294</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BLS1</td>
      <td>2007</td>
      <td>4.2755</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>6353.5815</td>
      <td>3.3295</td>
    </tr>
    <tr>
      <th>29</th>
      <td>RCI.B</td>
      <td>2007</td>
      <td>68.3509</td>
      <td>34542.6016</td>
      <td>3.90</td>
      <td>30668.1426</td>
      <td>7.3387</td>
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
    </tr>
    <tr>
      <th>690</th>
      <td>PVG</td>
      <td>2018</td>
      <td>4.7297</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>1985.1681</td>
      <td>1.7643</td>
    </tr>
    <tr>
      <th>691</th>
      <td>VET</td>
      <td>2018</td>
      <td>1.8182</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>6315.7832</td>
      <td>2.4468</td>
    </tr>
    <tr>
      <th>692</th>
      <td>DSG</td>
      <td>2018</td>
      <td>29.9487</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>3501.9514</td>
      <td>5.2637</td>
    </tr>
    <tr>
      <th>693</th>
      <td>ECA</td>
      <td>2018</td>
      <td>48.3262</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>16524.0566</td>
      <td>1.9500</td>
    </tr>
    <tr>
      <th>694</th>
      <td>BHC</td>
      <td>2018</td>
      <td>79.5116</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>10510.0762</td>
      <td>2.3301</td>
    </tr>
    <tr>
      <th>695</th>
      <td>KL</td>
      <td>2018</td>
      <td>49.9078</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>5151.8237</td>
      <td>3.2756</td>
    </tr>
    <tr>
      <th>696</th>
      <td>MEG</td>
      <td>2018</td>
      <td>64.471</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>2445.5652</td>
      <td>0.6197</td>
    </tr>
    <tr>
      <th>697</th>
      <td>FSV</td>
      <td>2018</td>
      <td>28.3481</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>3866.5066</td>
      <td>14.0518</td>
    </tr>
    <tr>
      <th>698</th>
      <td>ERF</td>
      <td>2018</td>
      <td>45.5365</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>3959.0337</td>
      <td>2.3204</td>
    </tr>
    <tr>
      <th>699</th>
      <td>BAD</td>
      <td>2018</td>
      <td>0.3112</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>1076.3000</td>
      <td>3.2235</td>
    </tr>
    <tr>
      <th>700</th>
      <td>CIGI</td>
      <td>2018</td>
      <td>63.8897</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>4030.4082</td>
      <td>9.7911</td>
    </tr>
    <tr>
      <th>701</th>
      <td>OSB</td>
      <td>2018</td>
      <td>15.4884</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>4303.6250</td>
      <td>3.6446</td>
    </tr>
    <tr>
      <th>702</th>
      <td>ECI</td>
      <td>2018</td>
      <td>38.1862</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>3107.1428</td>
      <td>5.3444</td>
    </tr>
    <tr>
      <th>703</th>
      <td>BB</td>
      <td>2018</td>
      <td>20.0519</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>7463.3804</td>
      <td>2.3713</td>
    </tr>
    <tr>
      <th>704</th>
      <td>MTY</td>
      <td>2018</td>
      <td>28.535</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>1520.9523</td>
      <td>2.5935</td>
    </tr>
    <tr>
      <th>705</th>
      <td>PSI</td>
      <td>2018</td>
      <td>18.9944</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>1819.0796</td>
      <td>5.0627</td>
    </tr>
    <tr>
      <th>706</th>
      <td>BBD.B</td>
      <td>2018</td>
      <td>72.4</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>10419.2314</td>
      <td>331.5385</td>
    </tr>
    <tr>
      <th>707</th>
      <td>GC</td>
      <td>2018</td>
      <td>35.7122</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>2820.1333</td>
      <td>5.1500</td>
    </tr>
    <tr>
      <th>708</th>
      <td>BAM.A</td>
      <td>2018</td>
      <td>12.8973</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>55326.3750</td>
      <td>1.6888</td>
    </tr>
    <tr>
      <th>709</th>
      <td>WCP</td>
      <td>2018</td>
      <td>-10.1336</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>3369.1038</td>
      <td>1.0425</td>
    </tr>
    <tr>
      <th>710</th>
      <td>SHOP</td>
      <td>2018</td>
      <td>37.5797</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>17794.7813</td>
      <td>9.3487</td>
    </tr>
    <tr>
      <th>711</th>
      <td>PD</td>
      <td>2018</td>
      <td>60.8833</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>1497.9512</td>
      <td>0.8482</td>
    </tr>
    <tr>
      <th>712</th>
      <td>RNW</td>
      <td>2018</td>
      <td>-14.8592</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>3173.7849</td>
      <td>1.3884</td>
    </tr>
    <tr>
      <th>713</th>
      <td>VII</td>
      <td>2018</td>
      <td>-19.2004</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>5555.5166</td>
      <td>1.2358</td>
    </tr>
    <tr>
      <th>714</th>
      <td>BCB</td>
      <td>2018</td>
      <td>7.2219</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>2836.1016</td>
      <td>1.7670</td>
    </tr>
    <tr>
      <th>715</th>
      <td>TCL.A</td>
      <td>2018</td>
      <td>30.8231</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>2792.2144</td>
      <td>1.9074</td>
    </tr>
    <tr>
      <th>716</th>
      <td>CP</td>
      <td>2018</td>
      <td>41.264</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>39135.0820</td>
      <td>5.9499</td>
    </tr>
    <tr>
      <th>717</th>
      <td>CNR</td>
      <td>2018</td>
      <td>14.6611</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>85207.3281</td>
      <td>4.9033</td>
    </tr>
    <tr>
      <th>718</th>
      <td>TRQ</td>
      <td>2018</td>
      <td>-28.0285</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>6097.3125</td>
      <td>0.5099</td>
    </tr>
    <tr>
      <th>719</th>
      <td>CCO</td>
      <td>2018</td>
      <td>8.3067</td>
      <td>55229.8008</td>
      <td>1.52</td>
      <td>5366.9531</td>
      <td>1.1095</td>
    </tr>
  </tbody>
</table>
<p>720 rows × 7 columns</p>
</div>




```python
###Fill in MktPrem
#This part should only be done once
dfA = fillMktPrem (dfA, start, end)
dfC = fillMktPrem (dfC, start, end)
```


```python
###Continue Cleanup
#Drop First Year
dfA = dfA.loc[dfA['Year'] != start]
dfC = dfC.loc[dfC['Year'] != start]

dfA = dfA.reset_index(drop = True)
dfC = dfC.reset_index(drop = True)
#Convert all inputs used to numeric

dfA.iloc[:, 2:] = dfA.iloc[:, 2:].convert_objects(convert_numeric=True)
dfC.iloc[:, 2:] = dfC.iloc[:, 2:].convert_objects(convert_numeric=True)
```

    C:\Users\Ry\Anaconda3\lib\site-packages\ipykernel_launcher.py:10: FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\Ry\Anaconda3\lib\site-packages\ipykernel_launcher.py:11: FutureWarning: convert_objects is deprecated.  Use the data-type specific converters pd.to_datetime, pd.to_timedelta and pd.to_numeric.
      # This is added back by InteractiveShellApp.init_path()
    


```python
dfA.head(30)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Year</th>
      <th>Return</th>
      <th>TRI</th>
      <th>TB90</th>
      <th>MKTCAP</th>
      <th>PB</th>
      <th>MKtReturn</th>
      <th>MktPrem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ATA</td>
      <td>2008</td>
      <td>39.0400</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>671.5371</td>
      <td>1.4624</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BIR</td>
      <td>2008</td>
      <td>219.2500</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1435.1438</td>
      <td>2.9372</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IOL</td>
      <td>2008</td>
      <td>-11.5591</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1136.4976</td>
      <td>5.7316</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CR</td>
      <td>2008</td>
      <td>110.6742</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1067.9100</td>
      <td>1.6085</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HPX</td>
      <td>2008</td>
      <td>28.5444</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>922.6240</td>
      <td>1.3032</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TOG1</td>
      <td>2008</td>
      <td>183.8667</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>2386.6516</td>
      <td>2.1461</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AGU</td>
      <td>2008</td>
      <td>86.1347</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>14185.3096</td>
      <td>3.4758</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PMCS</td>
      <td>2008</td>
      <td>17.8746</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1998.7028</td>
      <td>2.5971</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PXE</td>
      <td>2008</td>
      <td>50.8172</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1178.0902</td>
      <td>2.9557</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NVA</td>
      <td>2008</td>
      <td>13.6364</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1186.5299</td>
      <td>1.6271</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NTR</td>
      <td>2008</td>
      <td>97.4353</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>56345.7031</td>
      <td>8.1750</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RUS</td>
      <td>2008</td>
      <td>3.0191</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1942.6224</td>
      <td>2.1252</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BBD.B</td>
      <td>2008</td>
      <td>33.0645</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>14468.5039</td>
      <td>4.8851</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>13</th>
      <td>TLM</td>
      <td>2008</td>
      <td>3.5872</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>19121.1309</td>
      <td>2.1333</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CFP</td>
      <td>2008</td>
      <td>-12.1359</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1548.5165</td>
      <td>0.8747</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AGI</td>
      <td>2008</td>
      <td>18.8929</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1182.5964</td>
      <td>1.8471</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>16</th>
      <td>TESO</td>
      <td>2008</td>
      <td>24.8582</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1334.7764</td>
      <td>3.7403</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TXP</td>
      <td>2008</td>
      <td>50.1761</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>3877.1294</td>
      <td>5.4579</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GEA</td>
      <td>2008</td>
      <td>105.2811</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1284.9156</td>
      <td>3.4405</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GNA</td>
      <td>2008</td>
      <td>21.9512</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>6491.3252</td>
      <td>1.4825</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BNK</td>
      <td>2008</td>
      <td>304.1667</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>885.1299</td>
      <td>7.5475</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>21</th>
      <td>STE</td>
      <td>2008</td>
      <td>92.6241</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1180.5229</td>
      <td>2.3871</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CLS</td>
      <td>2008</td>
      <td>42.9253</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1793.5140</td>
      <td>0.8929</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ARE</td>
      <td>2008</td>
      <td>30.2488</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>851.7207</td>
      <td>2.5575</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CMT</td>
      <td>2008</td>
      <td>-6.4182</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>1177.5142</td>
      <td>1.3477</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>25</th>
      <td>CLL</td>
      <td>2008</td>
      <td>12.3288</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>865.2106</td>
      <td>1.8701</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CNQ</td>
      <td>2008</td>
      <td>25.5576</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>49019.0195</td>
      <td>3.6022</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CSU</td>
      <td>2008</td>
      <td>13.4179</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>498.6680</td>
      <td>6.6142</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>28</th>
      <td>HSE</td>
      <td>2008</td>
      <td>21.1355</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>39861.0703</td>
      <td>3.0938</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NXY</td>
      <td>2008</td>
      <td>12.6989</td>
      <td>34992.8008</td>
      <td>2.4</td>
      <td>17655.6055</td>
      <td>2.6526</td>
      <td>1.303316</td>
      <td>-1.096684</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create Fama French 3 factor model for Aggressive Strategy
FFA = pd.DataFrame(columns =
                  ["Year",
                   "Return",
                   "Rf",
                   "MktPrem",
                   "SMB",
                   "HML"                    
                  ])
FFAIndex = 0 
for i in range(start+1, end+1):
    FFA.loc[FFAIndex] = [i, 
                    calcReturn(dfA.loc[dfA['Year'] == i]), 
                    calcRf(dfA.loc[dfA['Year'] == i]), 
                    calcMktPrem(dfA.loc[dfA['Year'] == i]), 
                    calcSMB(dfA.loc[dfA['Year'] == i]), 
                    calcHML(dfA.loc[dfA['Year'] == i])
                   ]
    FFAIndex += 1
FFA['Year'] = FFA['Year'].astype(int)
```


```python
#Create Fama French 3 factor model for Conservative Strategy
FFC = pd.DataFrame(columns =
                  ["Year",
                   "Return",
                   "Rf",
                   "MktPrem",
                   "SMB",
                   "HML"                    
                  ])
FFCIndex = 0 
for i in range(start+1, end+1):
    FFC.loc[FFCIndex] = [i, 
                    calcReturn(dfC.loc[dfC['Year'] == i]), 
                    calcRf(dfC.loc[dfC['Year'] == i]), 
                    calcMktPrem(dfC.loc[dfC['Year'] == i]), 
                    calcSMB(dfC.loc[dfC['Year'] == i]), 
                    calcHML(dfC.loc[dfC['Year'] == i])
                   ]
    FFCIndex += 1
FFC['Year'] = FFC['Year'].astype(int)
```


```python
FFA
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Return</th>
      <th>Rf</th>
      <th>MktPrem</th>
      <th>SMB</th>
      <th>HML</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>36.8584</td>
      <td>2.40</td>
      <td>-1.0967</td>
      <td>18.7416</td>
      <td>43.6863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009</td>
      <td>12.8186</td>
      <td>0.20</td>
      <td>-17.5444</td>
      <td>12.4716</td>
      <td>36.1373</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>68.1621</td>
      <td>0.67</td>
      <td>10.8205</td>
      <td>14.7011</td>
      <td>60.3519</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>52.0113</td>
      <td>0.89</td>
      <td>8.2170</td>
      <td>24.8434</td>
      <td>61.1273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>29.9158</td>
      <td>1.03</td>
      <td>-5.2575</td>
      <td>5.4930</td>
      <td>46.9944</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2013</td>
      <td>38.0250</td>
      <td>0.99</td>
      <td>9.0373</td>
      <td>13.0665</td>
      <td>10.3304</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2014</td>
      <td>54.0386</td>
      <td>0.94</td>
      <td>26.1894</td>
      <td>25.7608</td>
      <td>66.1083</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015</td>
      <td>20.5506</td>
      <td>0.37</td>
      <td>-9.0489</td>
      <td>-17.1839</td>
      <td>61.2438</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016</td>
      <td>74.2083</td>
      <td>0.50</td>
      <td>8.1886</td>
      <td>26.1487</td>
      <td>18.0131</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017</td>
      <td>31.4535</td>
      <td>0.71</td>
      <td>6.5242</td>
      <td>-0.7788</td>
      <td>21.5734</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>34.1188</td>
      <td>1.52</td>
      <td>8.5722</td>
      <td>6.5659</td>
      <td>25.0311</td>
    </tr>
  </tbody>
</table>
</div>




```python
FFC
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Return</th>
      <th>Rf</th>
      <th>MktPrem</th>
      <th>SMB</th>
      <th>HML</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>-9.2666</td>
      <td>2.40</td>
      <td>-1.0967</td>
      <td>4.2731</td>
      <td>21.5956</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009</td>
      <td>-5.4656</td>
      <td>0.20</td>
      <td>-17.5444</td>
      <td>-1.4298</td>
      <td>-2.2787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>15.8135</td>
      <td>0.67</td>
      <td>10.8205</td>
      <td>11.8667</td>
      <td>-8.5265</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>8.0557</td>
      <td>0.89</td>
      <td>8.2170</td>
      <td>13.6852</td>
      <td>13.4448</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>5.7244</td>
      <td>1.03</td>
      <td>-5.2575</td>
      <td>4.1007</td>
      <td>5.9682</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2013</td>
      <td>20.0063</td>
      <td>0.99</td>
      <td>9.0373</td>
      <td>7.6418</td>
      <td>-6.0691</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2014</td>
      <td>17.3592</td>
      <td>0.94</td>
      <td>26.1894</td>
      <td>-3.0535</td>
      <td>3.7997</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015</td>
      <td>-1.1869</td>
      <td>0.37</td>
      <td>-9.0489</td>
      <td>-1.8108</td>
      <td>23.3356</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016</td>
      <td>4.3359</td>
      <td>0.50</td>
      <td>8.1886</td>
      <td>-6.1322</td>
      <td>4.7550</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017</td>
      <td>6.4929</td>
      <td>0.71</td>
      <td>6.5242</td>
      <td>-0.8448</td>
      <td>-4.3763</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>5.8371</td>
      <td>1.52</td>
      <td>8.5722</td>
      <td>-13.3971</td>
      <td>19.1939</td>
    </tr>
  </tbody>
</table>
</div>



## 3) Regression
```python
#Set up regression
Y = FFA.Return.values 
X = FFA[["MktPrem","SMB","HML"]]


model = sm.OLS( Y.astype(float), X.astype(float) )

result = model.fit()
print (result.params)
print(result.summary())
```

    MktPrem    0.662782
    SMB        0.994160
    HML        0.545028
    dtype: float64
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.877
    Model:                            OLS   Adj. R-squared:                  0.831
    Method:                 Least Squares   F-statistic:                     19.08
    Date:                Tue, 25 Sep 2018   Prob (F-statistic):           0.000528
    Time:                        11:42:09   Log-Likelihood:                -45.925
    No. Observations:                  11   AIC:                             97.85
    Df Residuals:                       8   BIC:                             99.04
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    MktPrem        0.6628      0.577      1.148      0.284      -0.669       1.994
    SMB            0.9942      0.484      2.054      0.074      -0.122       2.110
    HML            0.5450      0.158      3.443      0.009       0.180       0.910
    ==============================================================================
    Omnibus:                        0.240   Durbin-Watson:                   1.821
    Prob(Omnibus):                  0.887   Jarque-Bera (JB):                0.081
    Skew:                          -0.128   Prob(JB):                        0.960
    Kurtosis:                       2.666   Cond. No.                         5.55
    ==============================================================================
    

```python
#Set up regression
Y = FFC.Return.values 
X = FFC[["MktPrem","SMB","HML"]]
X = sm.add_constant(X)
X.rename(columns = {"const":"Intercept"}, inplace = True)


model = sm.OLS( Y.astype(float), X.astype(float) )

result = model.fit()
print (result.params)
print(result.summary())

```

    Intercept    5.810956
    MktPrem      0.528004
    SMB          0.151975
    HML         -0.310956
    dtype: float64
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.761
    Model:                            OLS   Adj. R-squared:                  0.659
    Method:                 Least Squares   F-statistic:                     7.436
    Date:                Tue, 25 Sep 2018   Prob (F-statistic):             0.0140
    Time:                        11:42:09   Log-Likelihood:                -31.609
    No. Observations:                  11   AIC:                             71.22
    Df Residuals:                       7   BIC:                             72.81
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      5.8110      2.143      2.711      0.030       0.743      10.879
    MktPrem        0.5280      0.148      3.562      0.009       0.177       0.879
    SMB            0.1520      0.224      0.678      0.519      -0.378       0.682
    HML           -0.3110      0.159     -1.953      0.092      -0.687       0.065
    ==============================================================================
    Omnibus:                        0.221   Durbin-Watson:                   1.382
    Prob(Omnibus):                  0.895   Jarque-Bera (JB):                0.364
    Skew:                          -0.238   Prob(JB):                        0.834
    Kurtosis:                       2.247   Cond. No.                         17.0
    ==============================================================================
    
   
