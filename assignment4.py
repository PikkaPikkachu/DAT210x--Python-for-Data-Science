import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')[0]
# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.
#
# .. your code here ..append

df.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']
# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics
#
# .. your code here ..
df.dropna(axis = 0, thresh = (len(df.columns) - 4), inplace = True)


# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..
df.drop([ 1, 13, 25, 37], axis = 0, inplace = True)

# TODO: Get rid of the 'RK' column
#
# .. your code here ..
df.drop(['RK'], axis = 1 , inplace = True)


# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..
df.reset_index(inplace = True)
#print df

# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
#
# .. your code here ..
df.GP = pd.to_numeric(df.GP)
df.G = pd.to_numeric(df.G)
df.A = pd.to_numeric(df.A)
df.PTS = pd.to_numeric(df.PTS)
df['+/-'] = pd.to_numeric(df['+/-'])
df.PIM = pd.to_numeric(df.PIM)
df['PTS/G'] = pd.to_numeric(df['PTS/G'])
df.SOG = pd.to_numeric(df.SOG)
df.PCT = pd.to_numeric(df.PCT)
df.GWG = pd.to_numeric(df.GWG)
df.PPG = pd.to_numeric(df.PPG)
df.PPA = pd.to_numeric(df.PPA)
df.SHG = pd.to_numeric(df.SHG)
df.SHA = pd.to_numeric(df.SHA)
#print df



# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#
# .. your code here ..
some = df.PCT.unique()
print len(some)