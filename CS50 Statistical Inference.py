#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #we import the data into python with the help of the pandas library
import numpy as np #import numpy library for future calculations
import matplotlib.pyplot as plt #we import matplotlib for generation of figures
import scipy.stats as stats #we import stats for the compuation of statistical calculations
df = pd.read_csv("salaries-by-region.csv") #the data is read into python and saved in the variable df
df["start_med"] = df["start_med"].str.replace("$","").str.replace(",","").astype(float) #the neccessary column is
                                                                                        #converted from string to float
df["mid_med"] = df["mid_med"].str.replace("$","").str.replace(",","").astype(float) #the neccessary column is
                                                                                        #converted from string to float
df.head(5) #shows the first 5 rows of the databased


# In[2]:


df["start_med"].describe() #gives some of the summary statistics for the start career median salary


# In[3]:


df["mid_med"].describe() #gives some of the summary statistics for the mid career median salary


# In[23]:


def more_stats(lst): # this function computes the rest of the descriptive stats(median and mode)
    lst = list(lst) #converts the argument inputted into the function into a list incase it was previously a dataframe
    mode = stats.mode(lst)[0][0] #computes the mode of the list
    median = np.median(lst) #computes the median of the list
    print("- Mode:", mode) #prints out the mode
    print("- Median:",median) #prints out the median
    return
print("Starting Career Median Salary") #summray stats for the start career median salary
more_stats(df["start_med"]) #function is called and take the start career median salary as argument
print("Mid-Career Median Salary") #summray stats for the mid career median salary
more_stats(df["mid_med"]) #function is called and take the mid career median salary as argument


# In[5]:


df["region"].describe() #gives some of the summary statistics for the region 


# In[6]:


#this code is used to query the dataframe for the exact data we require
northeast = df["start_med"][df["region"] == "Northeastern"] #saves the values of all start career median salaries of 
                                                            #in the Northeastern region into the variale northeast
midwest = df["start_med"][df["region"] == "Midwestern"]#saves the values of all start career median salaries of 
                                                       #in the Midwestern region into the variale midwest


# In[7]:


def summary_stats(lst):#This function computes the summary statistics relevant to the significance test of the 
                       #difference of mean
    lst = list(lst) #converts the argument inputted into the function into a list incase it was previously a dataframe
    print("- n:",len(lst)) #prints out the number of items in the list
    print("- mean:",np.mean(lst)) #prints out the mean of the list
    print("- SD:", np.std(lst,ddof = 1),"\n\n") #prints out the standard deviation of the list which Bessel's corrction 
                                                #has been applied to
    return
print("Start Salary for Northeastern Schools:\n")#displays summary statistics for the Northeastern Region
summary_stats(northeast)
print("Start Salary for Midwestern Schools:\n")#displays summary statistics for the Midwestern Region
summary_stats(midwest)

#This code generates histograms showing the distribution of the samples(Northeastern and Midwestern Start career 
#median salaries)
plt.hist(northeast) #creates the histogram for Northeastern region
plt.title("Distribution of Start Median Salary of Schools in Northeastern Region",fontsize = 10)#creates title
plt.ylabel("Frequency")#y-axis label
plt.xlabel("Start Median Salary in dollars per year")#x-axis label
plt.show() #displays histogram
plt.hist(midwest) #creates the histogram for Midwestern region
plt.title("Distribution of Start Median Salary of Schools in Midwestern Region",fontsize = 10)#creates title
plt.ylabel("Frequency")#y-axis label
plt.xlabel("Start Median Salary in dollars per year")#x-axis label
plt.show() #displays histogram


# In[8]:


start_med = list(df["start_med"])
print("n:",len(start_med))
print("mean:",np.mean(start_med))
print("SD:",np.std(start_med,ddof = 1))


# In[9]:


#This code computes a 99% confidence interval for the mean of the start career median salaries in the US
def confident(lst):#function that calculates the 99% confidence interval for the start career median salaries in the US
                   #it takes in a list as its only argument
    SE = np.std(lst,ddof = 1)/(len(start_med)**0.5)#Calculates the standard error using the sample standard deviation 
                                                   #Note: Bessel's correction has been applied here
    t_score = stats.t.ppf(0.995,len(start_med) - 1)#Calculates the appropriate t-score of the upper limit
    upper = int(np.mean(lst) + (t_score * SE)) #using the general formular, calculates the upper limit
    lower = int(np.mean(lst) - (t_score * SE)) #using the general formular, calculates the lower limit
    return [lower,upper] #returns a list containing the upper and lower limits
print("99% confidence interval:",confident(start_med)) #calls the function to work on the 99% confidence interval
                                                       #for the start career median salary for the US


# In[10]:


#This code displays the distributions of the variables of interest
plt.hist(start_med) #creates histogram for the distribution of the start career median salary in the US
plt.title("Distribution of Start Median Salary of Schools in the US",fontsize = 11)#histogram title
plt.ylabel("Frequency") #y-axis label
plt.xlabel("Start Median Salary in dollars per year") #x-axis label
plt.show() #displays histogram

#This generates the bar chart representing the number of schools per region who took part in the survey
region = list(df["region"]) #saves the column for regions into a list called region
cali = 0 ; west = 0 ; mid_west = 0 ; south = 0 ; north_east = 0 #initializes all variable to 0
for item in region: #this code runs through all items in region and accumulates as it finds various regions in the US
    if item == "California":
        cali += 1
    if item == "Western":
        west += 1
    if item == "Midwestern":
        mid_west += 1
    if item == "Southern":
        south += 1
    if item == "Northeastern":
        north_east += 1
count = [cali,mid_west,south,west,north_east] #all the frequencies are stored in a list called count
place = ["California","Midwestern","Southern","Western","Northeastern"] #list containing the names of regions
plt.bar(place,count) #creates bar graph of the regions and number of universities which participated
plt.xlabel("Region/State") #x-axis label
plt.ylabel("Number of school responses per region")#y-axis label
plt.title("Number of schools who took part in the survey per region",fontsize = 10)#bar graph title
plt.show()#shows bar graph


# In[11]:


#p-value calculator - finds p-value from z-score or t-score 
import scipy.stats as stats
z_score = -1.96
t_score = -4.568
ddof = 71
p_from_z_score = stats.norm.cdf(z_score)
p_from_t_score = stats.t.cdf(t_score,ddof)
print("p-value from z-score:",p_from_z_score)
print("p-value from t-score:",p_from_t_score)


# In[12]:


#t-score or z-score from p-value
p_value = 0.975
ddof = 21612
t_score_from_p = stats.t.ppf(p_value,ddof)
z_score_from_p = stats.norm.ppf(p_value)
print("t-score from p-value:",t_score_from_p)
print("z-score from p-value:",z_score_from_p)


# In[13]:


def diff_means_test(lst1,lst2,tails):#function that calculates Hedge's g, T-score and p-value used 
                                     #in tests for practical and statistical significance
                                     #This function takes in 3 arguments: 2 samples and the number of tails of the test
    lst1 = list(lst1) #converts the 1st argument inputted into the function 
                      #into a list incase it was previously a dataframe
    lst2 = list(lst2)#converts the 2nd argument inputted into the function
                     #into a list incase it was previously a dataframe
    n1 = len(lst1) #computes the number of items in the 1st list
    n2 = len(lst2) #computes the number of items in the 2nd list
    x1 = np.mean(lst1) #computes the mean of the 1st list
    x2 = np.mean(lst2) #computes the mean of the 2nd list
    s1 = np.std(lst1,ddof = 1) #computes the standard deviation of the 1st list
                               #Note: Bessel's correction is used
    s2 = np.std(lst2,ddof = 1) #computes the standard deviation of the 2nd list
                               #Note: Bessel's correction is used
    SE = ( ( s1**2/n1 ) + ( s2**2/n2 ) )**0.5 #using the general formular for calculating the standard error
                                              #in a difference of means test, the standard error is computed.
    t_score = ( (x1-x2) - 0 )/SE #using the general formular, the t-score is generated
    p_value = tails * stats.t.cdf(0 - abs(t_score),min( (n1 - 1),(n2 - 1) ) ) #the p-value is computed using the 
                                                                              #conservative approach for degrees 
                                                                              #of freedom
    s_pooled = ( ( (n1 - 1) * s1**2 +(n2 - 1) * s2**2 ) / (n1 + n2 -2) )**0.5 #the pooled standard deviation is 
                                                                             #calculated for use in computing Cohen's d
    d = (x1 - x2)/s_pooled #Cohen's d is calculated using the formular for Cohen's d
    correction = 1 - ( 3/( 4 * (n1 + n2) - 9 ) ) #Hedge's correction
    g = d * correction #Hedge's correction is applied on Cohen's d making it Hedge's g
    print("T-score:",t_score) #T-score is outputed
    print("p-value:",p_value) #p-value is outputted
    print("Hedge's g:",g) #Hedge's g is outputted
    return


# In[14]:


diff_means_test(northeast,midwest,2) #the function is called to work using the northeastern and midwestern 
                                     #samples for a two-tailed hypothesis test


# In[15]:


def more_stats(lst):
    import numpy as np
    import scipy.stats as stats
    lst = list(lst)
    mode = stats.mode(lst)[0][0]
    median = np.median(lst)
    print("- Mode:", mode)
    print("- Median:",median)
    return
print("Starting Carrier Median Salary")
more_stats(start_med)


# In[16]:


def correlation(predictor,response):
    
    #This part of the code computes the value of the correlation coefficient, r.
    
    lst1 = df[predictor] #assigns the column of the predictor to the variable lst1
    lst2 = df[response] #assigns the column of the response to the variable lst2
    
    global n #allows n be called outside correlation funtion
    n = len(lst1) #sample size of lists
    x1 = np.mean(lst1) #computes the mean of the predictor variable
    x2 = np.mean(lst2) #computes the mean of the response variable
    global sd1,sd2 #allows sd1 and sd2 to be called outside correlation funtion
    sd1 = np.std(lst1,ddof = 1) #computes the standard deviation of the predictor variable
    sd2 = np.std(lst2,ddof = 1) #computes the standard deviation of the response variable
    
    SU1 = [] #empty list that will store values of predictor variable in standard units
    SU2 = [] #empty list that will store values of response variable in standard units
    
    for item in lst1: #This loop converts all values of the predictor variable to standard units and stores them in SU1
        standard_unit = ( item - x1 )/sd1
        SU1.append(standard_unit)
    for item in lst2: #This loop converts all values of the response variable to standard units and stores them in SU2
        standard_unit = ( item - x2 )/sd2
        SU2.append(standard_unit)
        
    SU1 = np.array(SU1) #converts SU1 to an array
    SU2 = np.array(SU2) #converts SU2 to an array
    
    product_of_SU = SU1 * SU2 #Creates variable that stores the product of each value in SU1 and SU2
    r = np.sum(product_of_SU)/(len(product_of_SU) - 1)#finds the sum of values in product_of_SU and divides that by n-1
    print("Correlation Coefficient(r):",round(r,4)) #shows the value of r
    
    
    
    
    
    
    #This part of the code generates the Regression Line.
    
    global b1 #allows b1 be called outside correlation funtion
    b1 = (sd2/sd1) * r #computes the slope of the regression line
    b0 = x2 - ( b1 * x1 ) #computes the intercept of the regression line
    
    y_hat = [] #stores predicted values of response according to the regression equation
    for item in lst1: #generates all predicted values of y using the regression equation and adds them to y_hat
        y = b0 + b1 * item
        y_hat.append(y)
    
    print("Regression Equation:",response,"=",str( round(b0,2) ),"+",str( round(b1,2) ),"*",predictor)
    
    
    
    
    
    
    #This part of the code computes the Coefficient of Determination, R-squared.
    
    y_bar = [] #empty list 
    for item in lst2: #for loop that fills with the mean of the response for every value in lst2
        y_bar.append(x2)
    
    y_hat = np.array(y_hat) #converts y_hat to an array
    yi = np.array(lst2) #stores lst2 in array form
    y_bar = np.array(y_bar) #converts y_bar to an array
    
    sqrt_error = ( yi - y_hat ) ** 2
    SSE = np.sum(sqrt_error) #computes the SSE 
    
    sqrt_total = ( yi - y_bar ) ** 2
    SSTO = np.sum(sqrt_total) #computes the SSTO
    
    global R_squared #allows R_squared to be called outside correlation funtion
    R_squared = 1 - (SSE/SSTO) #Using SSE and SSTO, computes the R-squared value
    print("Coefficient of Determination:",round(R_squared,4))
    
    
    
    
    
    
    #This part of the code generates graph to test the conditions
    
    #Graph for linearity showing outliers
    
    #scatter plot of data
    plt.scatter(lst1,lst2,alpha = 0.6) #plots each point of (x,y)
    plt.grid() #creates gridlines
    
    #outliers highlighted
    outliers_x = [] #empty list that will house x coordinates of outliers
    outliers_y = [] #empty list that will house y coordinates of outliers
    
    for x in range(len(lst2)): #searches for outliers using condition of being more that 2.5 standard units away
        if abs( ( lst1[x] - x1 )/sd1 ) > 2.5 or abs( ( lst2[x] - x2 )/sd2 ) > 2.5:
            outliers_x.append(lst1[x])
            outliers_y.append(lst2[x])
            
    plt.scatter(outliers_x,outliers_y,color = "red",alpha = 0.6) #plots outliers in the data as red 
    plt.title("Start versus Mid Career Salary in $ for all regions in the US",fontsize = 10) #displays title 
    plt.xlabel("Starting Career Median Salary($)") #displays label on the x-axis
    plt.ylabel("Mid Career Median Salary($)") #displays label on the y-axis
    plt.plot(lst1,y_hat,color = "red") #plots regression line
    plt.show() #displays plot
    
    
    
    
    
    
    #This part of the code creates the Residual Plot
    
    res = [] #empty list that will store each residual
    for i in range(len(lst1)):
        y_hat = b0 + b1*lst1[i]
        res_y = lst2[i] - y_hat
        res.append(res_y)
        
    y0 = [] #empty list that will house a bunch of zeros
    for i in list(range(int(min(lst1)),int(max(lst1)+1))): #adds a zero to y0 for every value in lst1
        y0.append(0)
    
    plt.scatter(lst1,res,color = "green",alpha = 0.6) #plots the residuals against the predictor variable
    plt.plot(list(range(int(min(lst1)),int(max(lst1)+1))),y0,color = "black",linestyle = "dashed") #line at 0
    plt.title("Residual plot for Mid Career Salary($)",fontsize = 12) #displays title
    plt.xlabel("Starting Career Median Salary($)") #displays label on the x-axis
    plt.ylabel("Residuals of Mid Career Median Salary($)") #displays label on the y-axis
    plt.grid() #displays gridline
    plt.show() #displays plots
    
    
    
    
    
    
    #This part of the code plots a histogram for the residuals
    
    plt.hist(res,bins = 15,width = 3190) #creates histogram
    plt.title("Distribution of Residuals of Mid Career Salaries($) for all US regions",fontsize = "10") #title
    plt.xlabel("Residuals of Mid Career Median Salary($)") #displays label on the x-axis
    plt.ylabel("Frequency") #displays label on the y-axis
    plt.show() #displays plot
    
correlation("start_med","mid_med")


# In[17]:


def sig_test(b1,B1,tails): #carries out a test of statistical significance on the slope, b1.
    SE_b1 = ( ( (1 - R_squared)/(n - 2) )**0.5 ) * ( sd2/sd1 ) #computes standard error for the slope estimate
    t_score = (b1 - B1)/SE_b1 #computes t-score
    p_value = tails * stats.t.cdf(0 - abs(t_score),n - 2) #computes p-value based on t-score and number of tails
    print("T-score:",round(t_score,2))
    print("p-value:",p_value)
sig_test(b1,0,2)


# In[18]:


def conf_int(b1,conf_level):
    SE_b1 = ( ( (1 - R_squared)/(n - 2) )**0.5 ) * ( sd2/sd1 ) #computes standard error for the slope estimate
    p_value = (conf_level + (100 - conf_level)/2)/100
    ddof = n - 2
    t_score = stats.t.ppf(p_value,ddof)
    lower = b1 - t_score * SE_b1
    upper = b1 + t_score * SE_b1
    return [round(lower,3),round(upper,3)]
conf_int(b1,95)


# In[ ]:




