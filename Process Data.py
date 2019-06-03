#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
np = pd.np


# In[ ]:


df = pd.read_csv('EncCase19_test', delim_whitespace=True, header=None)


# ## Align data files
# 
# The observations file and stress files have a divergence in the lat/lon numbers do to truncation/rounding of the floating point values.  So we take the lat/lon values from the observations file and replace the lat/lon values in the stress file with the observation numbers.

# In[ ]:


obs = pd.read_csv('alldata.csv')
obs = obs.append(obs.tail(1), ignore_index=True)
repeated = pd.concat([obs] * 360, ignore_index=True)
df[1] = repeated['lat']
df[3] = repeated['lon']


# ## Clean up the stress data
# 
# The imported data has repeated column labels treated as columns themselves.  So here we remove the label columns and transpose the first row of their values to become column headings on the remaining dataset.

# In[ ]:


data = df[df.columns[1::2]]
headings = df[df.columns[::2]].head(1)
headings = headings.apply(lambda x: x.str.replace('=', ''))
data.columns = headings.iloc[[0]].values[0]


# In[ ]:


df = data.drop_duplicates().copy()

# Clean up and allow the GC to handle datasets we no longer need
data = None
repeated = None


# ## Convert azimuth from clockwise to counter-clockwise and calc max stress

# In[ ]:


df['zetaCW'] = (360. - df['Degrees(zeta)'])
df['maxStress'] = df[['sigThetaKPA', 'sigPhiKPA']].max(axis=1)

df.loc[df.maxStress == df.sigPhiKPA, 'crackDir'] = df['zetaCW'] % 180
df.loc[df.maxStress == df.sigThetaKPA, 'crackDir'] = ((df['zetaCW'] + 90) % 180) % 360


# ## Read in subset observation file and constrain data set

# In[ ]:


obs = pd.read_csv('observations.csv', sep='\s*,\s*', engine='python')


# In[ ]:


merged = pd.merge(df, obs, left_on=['lonDeg', 'latDeg'], right_on=['W Lon', 'Gr Lat'], how='inner')
merged.sort_values(['lonDeg', 'latDeg', 'Degrees(meanMotion)'], inplace=True)


# ## Calculate intermediate values and PDF

# In[ ]:


merged['diff'] = merged['maxStress'] - merged.shift()['maxStress']
merged.loc[merged['Degrees(meanMotion)'] == 0, 
           'diff'] = merged['maxStress'] - merged.shift(periods=-359)['maxStress']
merged['pdf'] = merged['maxStress']
merged.loc[merged['diff'] < 0, 'pdf'] = 0
merged.loc[merged['maxStress'] < 0, 'pdf'] = 0


# ## Grouping and calculating area under the curve

# In[ ]:


def process_group(group):
    return (abs(np.trapz(group['pdf'], group['crackDir'])))


# In[ ]:


grouped = merged.groupby(['latDeg', 'lonDeg'])

areas = pd.DataFrame(grouped.apply(process_group))
areas.columns = ['area']

merged = pd.merge(merged, areas, on=['latDeg', 'lonDeg'])


# In[ ]:


merged['probability'] = merged['pdf'] / merged['area']


# ## Optional - Testing normalization

# In[ ]:


def process_norm_group(group):
    return (abs(np.trapz(group['probability'], group['crackDir'])))

grouped = merged.groupby(['latDeg', 'lonDeg'])
test = pd.DataFrame(grouped.apply(process_norm_group))

test.columns = ['val']
test.sort_values('val', ascending=True)


# ## Finding the interpolated probability for each group

# In[ ]:


def find_probability(group):
    sorted_group = group.sort_values('crackDir')
    azimuth = group['Degrees(azimuth)'].max()
    return np.interp(azimuth, xp=sorted_group['crackDir'], fp=sorted_group['probability'])


# In[ ]:


grouped = merged.groupby(['latDeg', 'lonDeg', 'Degrees(azimuth)'])

lat = []
lon = []
azimuth = []
probability = []

for name, group in grouped:
    sorted_group = group.sort_values('crackDir')
    lat.append(name[0])
    lon.append(name[1])
    azimuth.append(name[2])
    probability.append(np.interp(name[2], xp=sorted_group['crackDir'], fp=sorted_group['probability']))
    
result = pd.DataFrame({ 'lat': lat, 
                      'lon': lon,
                      'azimuth': azimuth,
                      'probability': probability})

display(result)

# The following code also accomplishes the same result as the loop above,
# but may be more challenging to consume due to how the columns are constructed

# result = pd.DataFrame(grouped.apply(find_probability))
# result.columns = ['prob']
# # result.loc[-68.837121]
# # result.sort_values('prob', ascending=False)
# result


# In[ ]:




