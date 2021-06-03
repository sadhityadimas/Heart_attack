# Heart_attack_EDA_and_predictive_ML_model

## Predicting Mobile Phone Price Range 
* the dataset was uploaded to kaggle by  Rashik Rahman
* link to dataset https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
* 13 Features, 303 Samples
* 54.5% Sample has high heart attack Risk, 45.5% low heart attack risk


## Library Used:
* Pandas
* Numpy
* Matplotlib
* Scikit Learn 
* Seaborn

## Features Used
* age : numerical
* sex : categorical
* cp (chest pain) : categorical (1: typical angina,  2: atypical angina,  3: non-anginal pain,  4: asymptomatic)
* trtbps (resting blood pressure) : numerical
* chol (cholesterol) : numerical
* fbs (fasting blood sugar) : categorical ((fasting blood sugar > 120 mg/dl) (1 = true; 0 = false))
* restecg (resting electrocardiographic results) :  categorical 
* thalachh (maximum heart rate achieve) : numerical
* exng (exercise induce angina) : categorical
* oldpeak (ST depression induced by exercise relative to rest) : numerical
* slp (slope, slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)) : categorical
* caa ( number of major vessels): categorical
* thal : categorical

## Data Cleaning
* Already clean and ready to use

## Original Dataset

|    |   age |   sex |   cp |   trtbps |   chol |   fbs |   restecg |   thalachh |   exng |   oldpeak |   slp |   caa |   thall |   output |
|---:|------:|------:|-----:|---------:|-------:|------:|----------:|-----------:|-------:|----------:|------:|------:|--------:|---------:|
|  0 |    63 |     1 |    3 |      145 |    233 |     1 |         0 |        150 |      0 |       2.3 |     0 |     0 |       1 |        1 |
|  1 |    37 |     1 |    2 |      130 |    250 |     0 |         1 |        187 |      0 |       3.5 |     0 |     0 |       2 |        1 |
|  2 |    41 |     0 |    1 |      130 |    204 |     0 |         0 |        172 |      0 |       1.4 |     2 |     0 |       2 |        1 |
|  3 |    56 |     1 |    1 |      120 |    236 |     0 |         1 |        178 |      0 |       0.8 |     2 |     0 |       2 |        1 |
|  4 |    57 |     0 |    0 |      120 |    354 |     0 |         1 |        163 |      1 |       0.6 |     2 |     0 |       2 |        1 |


## Exploratory Data Analysis

### Univariate Analysis
### Numerical Features

#### Descriptive Statistic

|       |      age |   trtbps |     chol |   thalachh |   oldpeak |
|:------|---------:|---------:|---------:|-----------:|----------:|
| count | 303      | 303      | 303      |   303      | 303       |
| mean  |  54.3663 | 131.624  | 246.264  |   149.647  |   1.0396  |
| std   |   9.0821 |  17.5381 |  51.8308 |    22.9052 |   1.16108 |
| min   |  29      |  94      | 126      |    71      |   0       |
| 25%   |  47.5    | 120      | 211      |   133.5    |   0       |
| 50%   |  55      | 130      | 240      |   153      |   0.8     |
| 75%   |  61      | 140      | 274.5    |   166      |   1.6     |
| max   |  77      | 200      | 564      |   202      |   6.2     |

##### age
    * mean of age is 54
    * minimum age is 29, so no children, teen or early adult data
    * maximum age is 77 years old
    * median is 55 years old, close to our mean, there are probably no outliers, 
    * but we still have to check for outliers with histogram or distribution plot later 
    * standard deviation is 9, it means if our dataset distribution is normal, 68% of age values wil be around 45-63 
    * years old and 95% of age values will be around 36-72years old

##### resting blood pressure
    * it seems only systolic is recorded
    * mean and median is close to each other 131 and 130, so probably no outliers
    * this considered as high value because normal people trtbps is below 120/80 Mmhg and above 90/60 Mmhg
    * but we have to check the correlation with age, because older people then to have higher trtbs
    * minimum trtbs value is 94, the minimum value recorded in dataset  (healthy value)
    * maximum value is 200, it's hypertension, and people who have this high blood pressure
    * will experience chest pain, headache, shortness of breath or blood in the urine
    * it is the the highest trtbps in our dataset
    * standard deviation is 17, with our means of 131 (assuming no outliers and bell-shaped distribution),
    * 68% of trtbps values will be around 114-148
      
#### cholesterol
    * mean and median is still pretty close at 241 and 246 , but there maybe small amounts of outliers
    * these numbers are not good numbers
    * Total cholesterol levels less than 200 milligrams per deciliter (mg/dL) are considered desirable for adults. 
    * A reading between 200 and 239 mg/dL is considered borderline high
    * and a reading of 240 mg/dL and above is considered high
    * minimum and maximum values are 126 and 564, 
    * 126 is borderline low (too low can lead to another type of health risk, but we wouldn't explain it further)
    * 564 is too high  (this might be sign of obesity , and can cause symptoms; angina and chest pain)
    * standard deviation is 51, assuming no outliers (removing or transform) and bell-shape distribution,
    * 68% of chol values is around 190-297
    
    
####  thalachh (maximum heart rate achieved)
     * mean and median 149-153, there maybe small amount of outliers
     * according to heart.org, good heart rate per person is vary depend on age (220-age)
     * standard deviation 22 (68% of data will be around 127-175)
     * min and max values are 71 and 202

#### oldpeak (ST depression induced by exercise relative to rest)
    * still have not idea what this feature means, but it is in our dataset
    * so this must be important value to determine the risk ofheart attack
    * we can understand this further later in bivariate analysis and features importance
    
#### Distribution Analysis (numerical features)

![](/images/num_hist.png)

![](/images/num_boxplot.png)

##### Numerical Data Distribution Analysis
    * Age and Thallach seems to follow normal distribution
    * Cholesterol has outliers and skewed to the right
    * Trtbps has few outliers
    * Oldpeak is highly skewed to the right with long tail
    
    * Since i will use KNN classifier later, it is necessary to transform these numerical features

### Categorical Features

![](/images/target_pie.png)

![](/images/cat_pie.png)

* 54.5% sample in the dataset has high chance of heart attack
* Sex : There are more Male than Female in our dataset
* Chest pain : 48% chest pain is caused by typical angina 
* Fbs : 85.9% sample has normal blood sugar level (<120mg/dl)
* Restecg : 48.5% has normal heartwave, 50.5 has ST-T wave abnormality
* Exng : 67.7% has Angina caused by other reasons than exercise


### Bivariate Analysis
### Numerical Features

![](/images/num_pairplot.png)

* positive correlation between age and cholesterol, looks like older people tend to have higher cholesterol

### Categorical Features

![](/images/cat_pairplot.png)


## Building Machine Learning Model

### Mutual Information Score

![](/images/mutual_information_score.png)

* Thall, caa and cp have the strongest relationship with our target
* fbs, trtbps, and age has low relationship with our target
* meanwhile sex is independen and doesn't show any relationship with our target

### Preprocessing

#### Normalize and Transform Features 
* Robust for scaling and PowerTransform to make numerical features have gaussian distribution
* after scaling and transform
![](/images/num_hist_scale_and_transform.png)

#### One Hot Encoding for categorical features
* 8 categorical features, all have less than 5 cardinality, good for one hot encoding

#### Dataset after prepocessing

|    |       age |     trtbps |       chol |   thalachh |   oldpeak |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |   10 |   11 |   12 |   13 |   14 |   15 |   16 |   17 |   18 |   19 |   20 |   21 |   22 |   23 |   24 |
|---:|----------:|-----------:|-----------:|-----------:|----------:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
|  0 |  0.955998 |  0.827039  | -0.14692   |  -0.106865 |  1.19836  |   0 |   1 |   0 |   0 |   0 |   1 |   0 |   1 |   1 |   0 |    0 |    1 |    0 |    1 |    0 |    0 |    1 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |    0 |
|  1 | -1.80246  |  0.0254799 |  0.203788  |   1.92911  |  1.65951  |   0 |   1 |   0 |   0 |   1 |   0 |   1 |   0 |   0 |   1 |    0 |    1 |    0 |    1 |    0 |    0 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |
|  2 | -1.42282  |  0.0254799 | -0.834289  |   1.01659  |  0.660704 |   1 |   0 |   0 |   1 |   0 |   0 |   1 |   0 |   1 |   0 |    0 |    1 |    0 |    0 |    0 |    1 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |
|  3 |  0.128752 | -0.628494  | -0.0820789 |   1.36873  |  0.106164 |   0 |   1 |   0 |   1 |   0 |   0 |   1 |   0 |   0 |   1 |    0 |    1 |    0 |    0 |    0 |    1 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |
|  4 |  0.242685 | -0.628494  |  1.83183   |   0.523215 | -0.141799 |   1 |   0 |   1 |   0 |   0 |   0 |   1 |   0 |   0 |   1 |    0 |    0 |    1 |    0 |    0 |    1 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |

#### Split the dataset (80:20)

### Building and Training the Model 
### KNN
* ML model used : KNeighborsClassifier(n_neighbors=10)
* Metric Accuracy score : 0.8524590163934426

### Random Forest Classifier
* ML model used : RandomForestClassifier(n_estimators = 100, random_state=42)  
* Metric Accuracy score :  0.8852459016393442

### Score Summary KNN VS Random Forest Classifier

|    |      KNN |   Random Forest Classifier |
|---:|---------:|---------------------------:|
|  0 | 0.852459 |                   0.885246 |



