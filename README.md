# Introduction
In January of 2023, I was introduced to the idea of collecting data to improve Army Fitness when the Data Warfare Company from Fort Liberty came to Fort Stewart to institute a readiness and lethality study. This study is currently on-going and consists of a one-day course where trainers teach functional movements and general nutrition as well as introducing soldiers to their programming. 

In this analysis, I leveraged multiple machine learning algorithms to analyze the 2024 CrossFit Open Data with the goal of providing insights that can help the hired fitness professionals further structure and prepare their programming. This program is still in the early stages of its implementation, so I sought out to see if there was anything that data shows that could assist as the Army continues to tackle H2F initiatives. 

# What is the CrossFit Open? 
So what is the CrossFit Open? 

The idea was built around the Hopper Model for fitness, which is the idea that he or she who is fittest would be able to perform well at any random physical task that might present itself. The current structure is that over the course of three weeks, individuals complete three workouts and submit their scores to be ranked against other athletes worldwide.

While the purpose of the Open is to find the fittest athletes in each region to move on to the next stage of the competition, many just use the Open is a way to check in on their fitness, staying motivated for the upcoming year, accomplish personal goals, and celebrate with the community of CrossFit. So summarizing the workouts from this year: 24.1 consisted of a whole bunch of dumbbell snatches and burpees over the dumbbell, 24.2 was a workout to complete as many rounds as possible in 20 minutes of a row, deadlifts, and double-unders with a jump rope, and 24.3 was a whole bunch of barbell thrusters, pull-ups, and even muscle-ups if you made it to that point in the workout. 

# About the Dataset 
So, why did I choose to analyze the CrossFit Open?

First, if you actually look at the workout programming this H2F study is instituting, it strongly mimics the type workout you will see in a CrossFit gym (minus the handstand walks & muscle-ups) 	

And second, CrossFit deeply has its roots with the military using this type of programming – Law enforcement, firefighters and Military were among the first to use these type of workouts for training. 

That being said, the biggest reason is for two words: DATA COLLECTION. Sharing scores is an engrained part of the CrossFit culture, even down to writing daily workout scores on the affiliate whiteboard. This provides folks with an additional layer of accountability, shared commitment to the Workout of the Day, and in my case, a lot of data to analyze. 

To be clear, the Army does not need to become the world’s biggest CrossFit affiliate. But there are lessons and practices that the CrossFit community has learned and refined that could be easily integrated into the Army’s new fitness culture.

I will say I considered doing analysis on the H2F study workout scores, as they have data available for download of all the test scores, however, in my opinion, there isn't currently enough input yet to be able to really provide insightful and lasting conclusions. I do, however, expect to see useable data for great analysis here shortly as the popularity and interactions with the initiative grows and service members complete more tests.

Unfortunately, the data for the CrossFit Open was not easily available for download, so I built a web-scraper in Python that downloaded all of the 2024 Open Leaderboard Scores as well as the associated Athlete Profiles into a useable format. 

The Open Leaderboard had mainly generic information about the Open – Mainly the athlete’s Overall ranking and associated scores in the each of the workouts. 

However, the athlete profiles had a lot of information that I was able to associate with their performance, including biometric information such as Age, Weight, Gender, and Height, but also their self-inputted Benchmark Personnel Records.

I will quickly note that I do perceive there to be a skewness in the data due to the fact that all data points besides the validated Open scores by judges were self-inputted, but since this is the same approach that the study is taking to collect their data points for tests, and since there was such a large amount of input in both Men and Women, I concluded that there is still meaningful insights to be pulled out of this data. 

# Athlete Representation 
<img width="435" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/e6919c2c-7331-4176-9cb6-7ee14ce6e703">

<img width="447" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/f52c566a-37d8-4e4d-ad21-f2296d580775">

This year, there were just over 300,000 athletes represented in the CrossFit Open, with about 55% being men and 45% women. 

Looking across the regions, North America makes up the bulk of the Athletes, trailed by Europe and then South America. 

Here is some initial analysis I conducted for each of the CrossFit affiliates. Here, I plotted the location of each affiliate in the United States and just some baseline statistics for each as I began conducting my initial exploration. 

# Athlete Biometric Identifiers 
There were three main Biometric Identifiers in this dataset and I used height and weight to generate a fourth, BMI. 

Looking at Age, the mean age for Males and Females was around the same age, 35. 

For weight, males had a higher weight on average coming in around 187 lbs and women 142 lbs. For Height men also coming in slightly taller at 5’ 10” vs the women at 5’ 5”. 

Although the mean age was a little higher than the mean Age of soldiers in the Army, I was pleased with the distributions between men and women because the normalcy of the biometric identifiers, especially in height and weight, is quite similar to the Army distribution. 

# Body Mass Index (BMI) and Performance 
Body Mass Index (or BMI for short) is just a measure used to assess an individual’s body weight relative to their height. It is a commonly used screening tool to categorize individuals into different weight categories, such as underweight, normal weight, over weight, and obese. 

Looking at men’s BMI, the Bulk of the men’s BMI participating in the Open falls into the 25-30 category, which BMI would actually classify as overweight. For reference, a 5’ 11” man who weighs 190 lbs has a BMI of 26.5.

Looking at the Men BMI Rankings Chart at the Bottom: The Category Rank column is the highest average across all of the Benchmark Scores. Overall the 25-30 category performed the best in the benchmarks and the Overall Open. 

However, as expected, the heavier BMI athletes were able to back squat and deadlift more while the lighter athletes in the 20-25 range had the fastest 5k and sprint times. 

So based on initial results, it appears that the closer and athlete gets to the mean BMI (around 27 for men), the higher they perform overall, but you can tailor your BMI to either lift heavier or run faster if that is your target goal. 

That being said, much more research is needed. BMI doesn’t directly measure body fat and doesn’t account for factors such as muscle mass and bone density, so it may not be an accurate indicator of health for everyone, especially for athletes with a lot of muscle mass. 

# Athlete Benchmark Identifiers 

<img width="497" alt="Screenshot 2024-05-01 at 2 42 46 PM" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/debdb4c5-16b1-435e-b46f-a60404a6e04a">

# Benchmark Ranking 
Looking at Benchmark Athlete Performances, I wanted to see which exercise was the most indicative of overall performance. In total, I conducted four tests to look at this problem from a variety of angles and to see if outputs were similar. 

All of the tests were performed to find the goal of which feature was most likely to put an Athlete in the Top 25%. The reason I looked at the Top 25% was for two reasons: 1. This shows excellence in overall performance 
2. This year, the Top 25% was the percentage of athletes that made it to the next round of the games – the Quarter Finals

## Top 5% Benchmark Percentage 
<img width="149" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/71ae6d4b-3fbf-4a98-9672-569f98495e3f">

<img width="157" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/242d6a5a-a848-455f-8e86-4b4d76487887">

Looking at the percentage chart, the first test I conducted was.... for the top 5% of performances in a particular exercise, what percent of those Athletes made it to Quarter Finals? This is a good initial representation, but out of the three tests, is probably the least predictive because it takes into account the bias of whether or not Athletes inputted a PR for that particular exercise. 

## Hypothesis Tests
<img width="322" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/6db4ebde-d80a-44e2-ad56-a144a73d6109">

<img width="329" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/c0a03339-5290-497a-922d-c6d362c78e28">

The next test, represented by the bar charts, was a hypothesis test. In this test, I wanted to see if the Overall Rank in the CrossFit Open for the Top 25% of performances in an exercise was significantly different than the Overall Rank for the Bottom 75% and then I ranked each Exercise according to how significantly different each one was. 

Looking at both tests.......The Clean and Jerk, Snatch, and the Deadlift appear to be the top predictors of Overall Performance for men and women across both of these tests. 

## XGBoost Feature Importance 
<img width="440" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/032996b1-1be9-4dd4-8a85-9239189b1124">

Extreme Gradient Boosting is a popular machine learning algorithm known for its efficiency in handling structured data with labels and feature importance refers to a technique used in this method to determine the contribution of each feature in the model’s decision-making process. Essentially, it measures how much each feature influences the predictions made by the model. 

## Logistic Regression 
<img width="476" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/76119c1d-5704-4e33-afe5-6314f6d37f26">

Logistic Regression is a statistical model used for binary classification tasks where the target variable (in this case whether or not an Athlete made it to quarterfinals) has only two possible outcomes or classes. 

Using both of these models, I was able to determine the features from each that had the highest impact on the outcome of the Model. 

Across both, the Olympic Lifts of Clean and Jerk and Snatch were once again among the highest constant predictors in the models, with deadlift actually slightly trailing in these two models. 

However, in both of these, Fran (Which for quick references is a 21-15-9 complex of Barbell Thrusters (squatting to the floor and pushing a barbell over your head) and pullups) and the 5k Run arose as important predictors of Overall Performance. 

# What are the Chances of Making it to QuarterFinals? 
<img width="307" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/bbc956e7-293b-4771-8e25-0ae3d0927ac7">

<img width="333" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/bb404f80-e615-4fdd-aa47-639384cea2a2">

<img width="307" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/6fe8f981-ef39-45ff-a8b9-5d77edfab59d">

<img width="311" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/b16ce3aa-7b3b-4648-8517-4993eb8c9515">

Then, using the Logistic Regression mentioned in the previous slide, I built a model with that achieved a 78% accuracy of predicting whether an Athlete will make to Quarter Finals.

While this was specifically tailored for the CrossFit games, my thought process was that if you can semi-accurately predict CrossFit, where the benchmark tests change every year, you can even better predict scores on the ACFT, or similar standardized tests in the Army. 

Looking at this slide, if an Athlete's output was the mean scores for each benchmark exercise, they only have a 22% chance of making it to Quarterfinals, but if they are in the 75th percentile across all benchmark exercises, they have an 83% chance of making it to quarterfinals. 

Looking at the Hypothetical scores, I said.. if I kept everything the same as the mean scores for Fight Gone Bad and Down on the Chart (so the CrossFit workouts and the sprint) about how much would each weight lift and 5k run need to improve by to get to a 75% chance of making the games? 

While these numbers could be increased or decreased in some areas to achieve the same result, a hypothetical scenario is that your back squat would need to improve by 51 lbs, deadlift by 75 lbs, clean and jerk by 64 lbs, the snatch by 45 lbs, and the 5k run time decreased by 3 min. and 37 sec to get to a 20 min 5k. 

# Summarized Findings
<img width="346" alt="image" src="https://github.com/cdmoseley/Galvanize_Capstone_Crossfit/assets/161170070/ebe04cc4-307a-4a75-8051-124269fe85d6">

## Overall 
1. Olympic Lifts (Clean and Jerk / Snatch) seem to be the highest predictors of performance in both Men and Women 
A. Risk of Injury? 
B. Alternate power related exercises 

2. Recommend supplementing workouts with Barbell Thrusters (Fran), Deadlifting, and the 5k Run for the greatest results 

3. BMI’s closer the mean appears to have the highest correlation with overall performance, but further study needed to gain more insight (especially without any information on Body Fat %) 

4. For CrossFit Athletes: To achieve a greater than 50% chance of making it to Quarter Finals, you need average scores above the 58h percentile across all benchmark stats

## Deep Dive 
So to summarize my findings: 

The Olympic Lifts of Clean and Jerk and the Snatch appear to be the best predictors of Overall Performance in both Men and Women. Normally, this would mean that I automatically recommend to include more of these types of exercises in daily programming, however an important factor to consider is the technicality of these movements and the risk of injury. I think these workouts can be incorporated more under the assumption that there is a trained and certified fitness trainer overseeing that workout. On a side note, the Army needs more trained and certified fitness leaders in units regardless. 

If there is not someone who can adequately teach and oversee these types of movement, some good alternate exercises for power to include might be dumbbell clean and presses, dumbbell snatches, kettlebell swings, medicine ball cleans, and the overhead press. 

As of now, there are not a lot of Clean and Jerks and Snatches in the daily programming, but my assumption is that they’ve taken the risk factor into account for this decision. 

Next, my recommendation from the data output is to supplement workouts with Barbell Thrusters, Deadlifts, and the 5k Run. This recommendation actually supports their current programming, because there is a decent amount of each in the tests built into the website. 

Third, a quick study of BMI appears to be that higher performance occurred in what is traditionally known as “Overweight BMI” and closer to that mean, but further study would be needed for this and body fat % would be an essential tool to include in this analysis. 

Lastly, for the CrossFit Athletes out there, to achieve a greater than 50% chance of making it to Quarter Finals, or for it to be more likely than not that you make it, you need average scores above the 58th  percentile across all benchmark stats and if you want a greater than 75% chance of making it, you need average scores above the 71st percentile in all benchmark stats. The good news is that you can track your percentile in each of the exercises by inputted it into the CrossFit Games app (example on the left side of the slide). 

# Future Study and Impact 
As you can probably guess, I am a huge proponent of the H2F study currently being conducted and am highly optimistic that as buy-in increases and more data is inputted, this will soon lead to more tailored insights for soldiers and leaders. I look forward to conducting some data analysis in my own time on this data in the future. 

As for impact, it gives leaders a comprehensive idea of where their soldiers stand in their physical fitness. I believe this would make the Army less reliant on the ACFT, which people continue to debate whether or not it is a good test of overall fitness, and as this approach builds, dashboards are created, and the idea becomes more mainstream, it allows units to collaboratively learn from other units successes and provide leaders the information they need to maximize their units physical training regimen. 























