
# coding: utf-8

# # NYC Flights 2013 Dataset Analysis

# In[305]:

#IPython is what you are using now to run the notebook
import IPython
print ("IPython version:      %6.6s (need at least 1.0)" % IPython.__version__)

# Numpy is a library for working with Arrays
import numpy as np
print ("Numpy version:        %6.6s (need at least 1.7.1)" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print ("SciPy version:        %6.6s (need at least 0.12.0)" % sp.__version__)

# Pandas makes working with data tables easier
import pandas as pd
print ("Pandas version:       %6.6s (need at least 0.11.0)" % pd.__version__)

# Module for plotting
import matplotlib
print ("Mapltolib version:    %6.6s (need at least 1.2.1)" % matplotlib.__version__)

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print ("Scikit-Learn version: %6.6s (need at least 0.13.1)" % sklearn.__version__)


# ## Performing Exploratory Data Analysis

# In[306]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[307]:

flights_df= pd.read_csv('flights.csv')


# In[308]:

print (flights_df.shape)
print (flights_df.columns)
print (flights_df.dtypes)


# In[309]:

flights_df.dest.unique()
flights_df.head(10)


# ##Some Specific questions that may arise from the dataset

# (a) How many flights were there from NYC airports to Seattle in 2013?

# In[310]:

#subsetting the dataframe based on location and origin, taking the count after subsetting
len(flights_df[(flights_df.dest=='SEA') & ((flights_df.origin == 'EWR') | (flights_df.origin == 'LGA') | (flights_df.origin == 'JFK'))])                                         


#  There were 3923 flights from NYC to Seattle in 2013.

# (b) How many airlines fly from NYC to Seattle?

# In[311]:

#subsetting the dataframe based on location and origin, taking the unique count of carriers after subsetting
len(flights_df[(flights_df.dest=='SEA') & ((flights_df.origin == 'EWR') | (flights_df.origin == 'LGA') | (flights_df.origin == 'JFK'))].carrier.unique())                                      


# There are 5 airlines flying from NYC to Seattle.

# (c) How many unique air planes fly from NYC to Seattle?

# In[312]:

#subsetting the dataframe based on location and origin, taking the unique count of tailnum after subsetting
len(flights_df[(flights_df.dest=='SEA') & ((flights_df.origin == 'EWR') | (flights_df.origin == 'LGA') | (flights_df.origin == 'JFK'))].tailnum.unique())                                      


# There are 936 unique air planes flying from NYC to Seattle.

# (d) What is the average arrival delay for flights from NC to Seattle?

# In[313]:

#subsetting the dataframe based on location and origin, taking the mean of arr_delay after subsetting (excluding non delayed flights)
flights_df_filtered = flights_df[(flights_df.arr_delay>0) & (flights_df.dest=='SEA') & ((flights_df.origin == 'EWR') | (flights_df.origin == 'LGA') | (flights_df.origin == 'JFK'))]     
print(np.mean(flights_df_filtered.arr_delay))

#subsetting the dataframe based on location and origin, taking the mean of arr_delay after subsetting (including non delayed flights)
flights_df_filtered = flights_df[(flights_df.dest=='SEA') & ((flights_df.origin == 'EWR') | (flights_df.origin == 'LGA') | (flights_df.origin == 'JFK'))]     
print (np.mean(flights_df_filtered.arr_delay))


# I have filtered out only the flights that had arrival delay at the Seattle airport. Thus to find the average arrival delay, I am not considering the flights that were on time or reached before time. Thus the arrival delay for flights from NYC to Seattle is 39.79984 minutes.
# 
# If we take all the flights landing at Seattle from NYC, then the average arrival delay decreases to -1.09 minutes.

# (e) What proportion of flights to Seattle come from each NYC airport?

# In[314]:

#Subsettign dataframe based on destination as Seattle
flights_seattle_bound_df = flights_df[(flights_df.dest == 'SEA')]  

#Grouping by origin and taking sum count of flights from individual origin locations
grouped_data = flights_seattle_bound_df.groupby([flights_seattle_bound_df.origin]).size()

#Dividing the sum count of flight from individual locations by total flights to find percentage
print (pd.DataFrame({'Percentage':grouped_data / len(flights_seattle_bound_df)}))


# Firstly, I have filtered the dataset to flights bound to Seattle destination. Then I have grouped the flights by their origin. To find the proportion of flights from each of the airport, using the number of flights from each airport, I have divided each by the total number of flights from NYC to Seattle which is the length of the filtered data frame. Thus there are 46.67% flights to Seattle are from EWR airport and 53.33% flights are from JFK airport flying out of NYC.

# Flights are often delayed. Let us explore some delay patterns
# 
# (a) Which date has the largest average departure delay? Which date has the largest average arrival delay?

# In[380]:

#Method 1
#Grouping flights df by month and day and caluclating mean directly
grouped_data = flights_df.groupby([flights_df.month,flights_df.day]).mean()
print(max(grouped_data.arr_delay))
print(max(grouped_data.dep_delay))



# Largest average departure delay is 83.53 minutes.
# Largest average arrival delay is 85.86 minutes.

# In[386]:

#Method 2
#Using pivot table function on flights df which automatically calculates the aggregated mean of a column (dep_delay)
#over the given set of columns (month, day). Sorting in descending order to find max value as the first values
pivot_df_dep_delay=flights_df.pivot_table(columns=['month', 'day'], values='dep_delay')
print(pivot_df_dep_delay.sort_values(ascending=False).head(1))

#Using pivot table function on flights df which automatically calculates the aggregated mean of a column (arr_delay)
#over the given set of columns (month, day). Sorting in descending order to find max value as the first values
pivot_df_arr_delay=flights_df.pivot_table(columns=['month', 'day'], values='arr_delay')
print(pivot_df_arr_delay.sort_values(ascending=False).head(1))


# In[ ]:

Largest average departure delay is 83.53 minutes.
Largest average arrival delay is 85.86 minutes.


# (b) What was the worst day to fly out of NYC in 2013 if you dislike delayed flights?
# 

# In[316]:

#Method 1
#Using Method 1 from above solution and subsetting the grouped data to find the max dep_delay value 
grouped_data = flights_df.groupby([flights_df.month,flights_df.day]).mean()
print(grouped_data[grouped_data.dep_delay == max(grouped_data.dep_delay)])


# The worst day to fly out of NYC in 2013 is 8th March as the this was the day with max average delay in departure.

# In[388]:

#Method 2
#Using pivot table function on flights df which automatically calculates the aggregated mean of a column (dep_delay)
#over the given set of columns (month, day). Sorting in descending order to find max value as the first values
pivot_df_dep_delay=flights_df.pivot_table(columns=['month', 'day'], values='dep_delay')
print(pivot_df_dep_delay.sort_values(ascending=False).head(1))


# Using this method as well 8th March 2013 is the worst day to fly if a person disliked delyed flights based on average delay in departure being the highest.

# (c) Are there any seasonal patterns in departure delays for flights from NYC?

# In[379]:

#Method 1
#grouping flights df to calculate mean departure delay over the months
grouped_data = pd.DataFrame(flights_df.groupby([flights_df.month])['dep_delay'].mean().reset_index(name='Mean_Delay_Departure'))
print(grouped_data)

#Plotting Mean Departure delay over a Day
plt.plot(grouped_data.month, grouped_data.Mean_Delay_Departure,'')
plt.xlim(1, 12)
plt.ylabel('Mean Delay Departure Time (minutes)')
plt.xlabel('Months')
plt.title('Mean Departure delay over a Day')
plt.show()


# We see a peculiar seasonal pattern. The delays are more during holiday seasons like the summers and end of year. This is tiem usually when people travel the most and hence due to more passenger traffic, we could expect delays at the airports.

# In[406]:

#Method 2
#Filtering flights df to capture only delyed flights over the whole population
filtered_flights_df = flights_df[flights_df.dep_delay > 0]

#Grouping by month to find the count of delays
grouped_data = pd.DataFrame(filtered_flights_df.groupby([filtered_flights_df.month])['dep_delay'].size().reset_index(name='Count'))

#Plotting Number of delays per Month
plt.bar(grouped_data.month, grouped_data.Count, color="red")
plt.ylabel('Number of Delays')
plt.xlabel('Months')
plt.title('Number of delays per Month')
plt.show()


# From the above visualization we can again see that number of delays are higher during the holiday season and lower during non- holiday seasons.

# (d) On average, how do departure delays vary over the course of a day?

# In[405]:

#Method 1
#Grouping flights df over hour to understand the variation over the course of a day, calculating mean departure delay
grouped_data = pd.DataFrame(flights_df.groupby([flights_df.hour])['dep_delay'].mean().reset_index(name='Mean_Delay_Departure'))
print(grouped_data)

plt.plot(grouped_data.hour, grouped_data.Mean_Delay_Departure,'')
plt.xlim(0, 24)
plt.ylabel('Mean Delay Departure Time (minutes)')
plt.xlabel('Hours during the day')
plt.title('Mean Departure delay over a Day')
plt.show()


# The departure delays are higher in the late night - early morning flights to the mid day flights. The reason for this could be more number of flights taking off during night times. 

# In[417]:

#Method 2
#Filtering flights df to capture only delayed flights over the whole population
filtered_flights_df = flights_df[flights_df.dep_delay > 0]

#Grouping by hour to find the count of delays
grouped_data = pd.DataFrame(filtered_flights_df.groupby([filtered_flights_df.hour])['dep_delay'].size().reset_index(name='Count'))

#Grouping by hour to find the total count of flights over a day to further find the percentage of delays
grouped_data_non_filtered = pd.DataFrame(flights_df.groupby([flights_df.hour])['dep_delay'].size().reset_index(name='Total_Count'))

#Merging both the dataframes
df_merged = pd.merge(grouped_data, grouped_data_non_filtered, on=['hour'])

df_merged['Percentage_Delays'] = (df_merged.Count/df_merged.Total_Count)*100
print(df_merged)
#Plotting Number of delays per Hour of a day
plt.bar(df_merged.hour, df_merged.Percentage_Delays, color="black")
plt.ylabel('Percentage of Delays')
plt.xlabel('Hours')
plt.title('Percentage of delays per Hour of a day')
plt.show()


# From above visualization we can see that 100% of flights are delayed for late night - early morning. Thus there is trend during the day. We can also gather from above that not many flights fly during late night - early morning times thus the reason for such delays is not known.

# Which flight departing NYC in 2013 flew the fastest?

# In[381]:

#Calculating speed based on distance travelled and air_time
flights_df['speed'] = flights_df['distance']/flights_df['air_time']

#Filtering based on max overall speed of plane in travelling to its destination
print(flights_df[(flights_df.speed == max(flights_df.speed))]['flight'])


# Flight number 1499 flew the fastest from NYC in 2013. 

# Which flights (i.e. carrier + flight + dest) happen every day? Where do they fly to?

# In[384]:

#Grouping by carrier + flight + dest and counting the number of such flights 
#thus we get the number of days the same carrier + flight + dest has happened
grouped_data = pd.DataFrame(flights_df.groupby([flights_df.carrier,flights_df.flight,flights_df.dest]).size().reset_index(name='Count'))

#Filtering on max value of the count to find all such combinations
grouped_data[grouped_data.Count == max(grouped_data.Count)]


# The above 18 different flights happen everyday. The location of the final destination is given by the dest column.

# Exploring through research questions
# 

# Research Question - Which carriers have been the top and the bottom performers in 2013?
# 
# Why this question?
# I think this quesion will help us identify the carriers which have been performing badly through out the year. By knowing this we can help the general public to avoid commuting by this carrier.
# 
# I feel that to answer this question we would have to look at the number of flights departing delayed and also arriving  delayed. I plan to ignore the flights which departed delayed though arrived on or before time, as in all the time was covered up while flying. Although there is a ethical promise that a carrier makes to start on scheduled time, I plan to ignore this concern in my below analysis.

# In[321]:

#Fitering dataset for flights having arr_delay>0 and dep_delay>0
flights_df_filtered_carrier = flights_df[(flights_df.arr_delay>0) & (flights_df.dep_delay>0)]  

#Grouping by carrier and getting the count
grouped_career_filtered = flights_df_filtered_carrier.groupby([flights_df_filtered_carrier.carrier]).size().reset_index(name='Size')
grouped_career_filtered

#Removing NA's from dep_time column
flights_df_filtered_total = flights_df[np.isfinite(flights_df['dep_time'])] 

#Grouping by carrier to find the total count for each carrier
grouped_career_total = flights_df_filtered_total.groupby([flights_df_filtered_total.carrier]).size().reset_index(name='Total_Size')
grouped_career_total

#Merging both the dataframes
df_col_merged = pd.merge(grouped_career_filtered, grouped_career_total, on=['carrier'])

#CalculatingPercentage delays
df_col_merged['Percentage_Delays'] = (df_col_merged.Size/df_col_merged.Total_Size)*100

df_col_merged

#Plotting Percent Delay by Carrier through 2013
ind = np.arange(len(df_col_merged.carrier))
plt.bar(ind, df_col_merged.Percentage_Delays, color="blue")
plt.ylabel('Delay Percentage')
plt.xlabel('Carriers')
plt.title('Percent Delay by Carrier through 2013')
plt.xticks(ind, df_col_merged.carrier)
plt.show()

#Grouping by carrier and getting the mean arrival delay
df_mean_arr_delay = flights_df_filtered_carrier.groupby([flights_df_filtered_carrier.carrier])['arr_delay'].mean().reset_index(name='Mean_Arrival_Delay')

#Plotting Average Arrival Delay for each Carrier
ind = np.arange(len(df_mean_arr_delay.carrier))
plt.bar(ind, df_mean_arr_delay.Mean_Arrival_Delay, color="green")
plt.ylabel('Avg. Arrival Delay (minutes)')
plt.xlabel('Carriers')
plt.title('Average Arrival Delay for each Carrier')
plt.xticks(ind, df_col_merged.carrier)
plt.show()


# Analysis - 
# The performance of the carrier can be gauged by (1) what percentage of flights of a particular carrier are delayed in departure and also delayed in arrival and (2) what is the average delay in arrival time for each of the carrier over the year of 2013.
# 
# Firstly, looking at the visualization (Percent Delay by Carrier through 2013), we observe that carrier FL has the highest delay %, thus making it the least performer among other carriers. Carrier HA has the best performance in terms of delay %.
# 
# Secondly, looking at the visualization (Average Arrival Delay for each Carrier), we observe that OO and HA have higher arrival delays among other carriers. UA and US carriers perform best when looking from this perspective. I have considered average arrival delay because I feel that in all for a traveller the delay in reaching a particular point is more significant than delay in departure.

# What weather conditions are associated with flight delays leaving NYC? Use graphics to explore.

# In[364]:

#Loading weather data
weather_df= pd.read_csv('weather.csv')

#Filtering only delayed flights from all airports
flights_df_filtered_delayed = flights_df[(flights_df.dep_delay>0)] 

#Grouping by origin, hour, day and month, as analysis would be at the granularity of the weather dataset
groupby_output = flights_df_filtered_delayed.groupby([flights_df_filtered_delayed.origin, flights_df_filtered_delayed.month, flights_df_filtered_delayed.day, flights_df_filtered_delayed.hour])
grouped_origin_time_hour = groupby_output['dep_delay'].agg([np.size, np.mean]).reset_index()


# Above, I have loaded the flights and weather dataset and also filtered data according to delayed or not delayed as that will help in comparison when combined with the weather datset. By filtering out the delayed flights, I plan to study average time delay and number of delays per some of the variables (visib, wind_speed, wind_gust) in the weather dataset. If we consider the whole dataset, without filtering, then due to averaging out we could miss out on some of the specific flights that were actually delayed because there are flights which have departed early. Thus to avoid such a miss, I have considered only delayed flights for analysis. Also some of the plane models might not be affected by weather and hence might takeoff on or before time, to remove those biases, I consider only delyed flights.
# 
# As the granularity of analysis would of the weather dataset, I have grouped the flight_delayed dataset by origin and time_hour bringing it to similar granularity. By grouping, I have calculated the average delay time at a particular time_hour and airport and also calculated total count of delays at a particular time_hour and airport.

# In[363]:

#Joining the above output with the weather dataset. This is an inner join and the hour-day-month for which 
#data is not present in weather dataset are omitted.
df_weather_flights_merged = pd.merge(grouped_origin_time_hour, weather_df, on=['origin','hour','day','month'])

#Renaming Columns
df_weather_flights_merged = df_weather_flights_merged.rename(columns={'size': 'Count', 'mean': 'TotalDelay'})


# Above, I have merged the weather and the grouped dataset so that it will help in analysis. The merging is done on origin and time_hour columns.

# In[385]:

#Working on the combined df, grouping by visibility to see trends between
#delays and the weather variables
by_visib = df_weather_flights_merged.groupby([df_weather_flights_merged.visib])['TotalDelay', 'Count'].agg([np.mean]).reset_index()
print(by_visib)

#plot the data
plt.scatter(by_visib.visib, by_visib.TotalDelay)
plt.xlim(0, 11)
plt.ylim(0, 110)
plt.ylabel('Average Departure Delay Time (minutes)')
plt.xlabel('Visibility (miles)')
plt.title('Visibility vs. Average Departure Delay')
plt.show()

#plot the data
plt.scatter(by_visib.visib, by_visib.Count)
plt.xlim(0, 11)
plt.ylim(0, 20)
plt.ylabel('Average Number of Delays')
plt.xlabel('Visibility (miles)')
plt.title('Visibility vs. Average Number of Delays')
plt.show()


# From the above graphics we can see that lower the visibility slightly higher are the average departure delay time and average count of number of delays. This proves that one of the weather variable like the visibility slightly impacts the flights from NYC.
# 
# To explore more below we can look at the impact of wind_speed on the flight delays

# In[365]:

#Working on the df_weather_flights_merged df, grouping by wind_speed to see trends between 
#delays and the weather variables
df_wind_analysis = df_weather_flights_merged.dropna(subset = ['wind_speed'])

#Calculating average departure delays vs. wind_speed
by_wind = df_wind_analysis.groupby([df_wind_analysis.wind_speed])['Count','TotalDelay'].agg([np.mean]).reset_index()
print(by_wind)

#Plotting the data
plt.scatter(by_wind.wind_speed, by_wind.TotalDelay)
plt.xlim(0, 40)
plt.ylim(0, 120)
plt.ylabel('Average Departure Delay Time (minutes)')
plt.xlabel('Wind Speed (mph)')
plt.title('Average Departure Delay vs. Wind Speed')
plt.show()


# Here, I have removed one observation from the graph as the information in the tuple seems to be an outlier. Values of wind_speed and wind_gust are 100 times greater than the other values in the same column. I have also omitted NA's from wind_speed column. I have grouped on wind_speed and calculated the average departure delay in minutes per value of wind speed (as there are only specific values of wind speed observed in the dataset, it actually is very much a continuous variable). The above graphic depicts that as the wind speed increases the average departure delay time increases. Thus wind_speed impacts flights from NYC.
