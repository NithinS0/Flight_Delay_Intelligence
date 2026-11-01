# Flight Delay Prediction and Statistical Intelligence System
## Complete Presentation Speaker Notes & Jargon Guide

> **Target Audience:** Academic examiners, capstone evaluation panels, and project review committees.  
> **Presentation Tone:** Technically rigorous, clear, confident, and student-friendly.

---

## Part 1: Technical Concepts & Jargon Explained Simply

Before delivering the presentation, review these plain-language explanations of the key data science and statistical concepts used throughout the project.

### 1. Classification vs. Regression
* **Classification:** An algorithm predicts a discrete category. In our project, classification answers a binary yes/no question: *"Will this flight experience a departure delay of 15 minutes or more (`DELAYED = 1`) or depart on time (`DELAYED = 0`)?"* (15 minutes is the formal FAA benchmark).
* **Regression:** An algorithm predicts a continuous quantity on a numerical scale. In our project, regression answers a measurement question: *"Exactly how many minutes (`DEP_DELAY`) will this flight be delayed?"* Regression is evaluated in minutes (MAE/RMSE), not as a percentage accuracy.

### 2. Data Leakage
* **Definition:** When information that would not be available at the time of prediction is accidentally fed into a machine learning model during training.
* **Why it Ruined Prior Studies:** In flight delay prediction, some researchers accidentally include features like `DEP_TIME` (actual departure time) or `TAXI_OUT` (taxi duration) to predict departure delay. That is leakage because you only know what time a plane pushed back *after* it pushed back!
* **Our Solution:** We enforce a strict zero-leakage policy: our models use only scheduled timetable details and historical statistics calculated strictly prior to the flight date ($t < \text{FL\_DATE}$).

### 3. Chronological Time-Series Splitting vs. Random K-Fold
* **Random K-Fold (Wrong for Time Series):** If you randomly shuffle flight data, the model might train on a flight from August 2023 and use it to "predict" a flight from March 2019. That cheats by leaking future trends into the past.
* **Chronological Splitting (Our Method):** We train on historical data (2019–2021), tune hyperparameters and decision thresholds on middle data (early 2022), and perform a single final evaluation on an untouched future test set (late 2022 to mid-2023). This mirrors genuine real-world deployment.

### 4. Natural Class Distribution vs. Synthetic Sampling (Why We Avoid SMOTE)
* **Natural Imbalance:** In real life, only about 18% of flights are delayed $\ge 15$ min, while 82% depart on time.
* **Why We Avoid SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically inventing fake delayed flights distorts real-world prior probabilities (base rates). When deployed on real flights, a SMOTE-trained model triggers massive false alarms because it was trained to expect a 50/50 world that does not exist. We preserve the true operational 18% base rate and handle class imbalance properly via **validation-tuned decision thresholds ($\tau^*$)**.

### 5. Decision Threshold Optimization ($\tau^*$)
* **Default Threshold (0.50):** Standard classifiers predict "Delayed" only if the predicted probability $P(\text{Delayed}) \ge 0.50$. But in an 18% base-rate world, a model will almost never output $\ge 0.50$, resulting in high precision but abysmal recall (missing 80% of delays).
* **Optimized Threshold ($\tau^* = 0.2372$):** By sweeping thresholds on our validation set, we find that classifying flights as delayed when probability exceeds $23.7\%$ maximizes the delayed-class F1-score, lifting disruption recall to nearly $60\%$ without flooding operators with false alarms.

### 6. Skewness, Kurtosis, and Robust L1 Loss
* **Skewness ($\gamma_1 = 11.49$):** The delay distribution is not a symmetrical bell curve. Most flights depart between $-10$ and $+5$ minutes, but a long tail stretches out to $+2,966$ minutes (extreme weather/mechanical groundings).
* **Excess Kurtosis ($\gamma_2 = 243.94$):** "Fat tails." Extreme outliers occur thousands of times more often than a normal Gaussian distribution would ever predict.
* **Why L1 Loss (MAE) Instead of L2 Loss (MSE):** L2 loss squares errors, so a single 2,000-minute delay has a penalty of $4,000,000$. This causes standard linear models to pull their predictions up to $+10$ minutes for every flight, ruining predictions on the 65% of flights that depart early. Formulating models with **L1 loss** directs them to predict the **conditional median** ($-2.0$ min), minimizing typical absolute error.

### 7. Evaluation Metrics Explained
* **Recall (Sensitivity):** Out of all flights that genuinely experienced a delay $\ge 15$ minutes, what percentage did our model successfully catch? (Our champion catches **$59.27\%$**).
* **Precision:** When our model sounds the alarm and predicts "Delayed", how often is it actually right? (Our champion achieves **$36.56\%$**, more than double the natural 17.7% random chance rate).
* **F1-Score:** The harmonic mean of precision and recall. It balances catching delays against avoiding false alarms.
* **ROC-AUC (0.7068):** The probability that the classifier will rank a randomly chosen delayed flight higher than a randomly chosen on-time flight. An AUC of 0.50 is coin-toss guessing; 1.0 is perfection.
* **MAE (Mean Absolute Error, 18.56 min):** The average absolute difference between predicted and actual delay in minutes.
* **Median Absolute Error (5.16 min):** The median error across all test flights. For over 50% of commercial flights, our model is accurate within ~5 minutes of actual pushback.

---

## Part 2: Slide-by-Slide Presentation Speaking Script

Use this script verbatim or adapt it naturally during your oral presentation.

---

### Slide 1: Title Slide
"Good morning, respected members of the panel. Today, we are proud to present our project titled: **Flight Delay Prediction and Statistical Intelligence System**. 

Commercial flight delays cost the aviation industry tens of billions of dollars annually and cause widespread disruption for passengers and air traffic control. In this project, we develop a comprehensive, leakage-free data science platform that combines rigorous inferential statistics with machine learning classification and regression across 3 million commercial flights. We will demonstrate how our system accurately predicts flight disruption risks and delay durations hours before scheduled departure."

---

### Slide 2: Problem Statement & Motivation
"To understand why this project is necessary, we must examine how airline operations actually work. At two hours before departure, an airline operations dispatcher needs to know two critical things:
First: *Is this flight going to be delayed by 15 minutes or more?* If yes, they can hold connecting flights, warn passengers, or reassign ground crews.
Second: *If it is delayed, by how many minutes?*

Unfortunately, when you review previous machine learning literature in this domain, many published models commit severe methodological errors. Some use post-departure variables like taxi-out time or arrival delays—information you cannot possibly know before pushback. Others use artificial sampling techniques like SMOTE that distort real-world probabilities. Our motivation was to build a scientifically rigorous, zero-leakage system that solves both the classification and regression challenges on untouched operational data."

---

### Slide 3: Project Aim & Objectives
"Our primary aim was to build an end-to-end statistical intelligence and predictive system. To achieve this, we executed six systematic objectives:
1. Establish strict data governance across 3 million BTS flight records, preserving genuine operational delays up to 2,966 minutes while eliminating cancellations and duplicates.
2. Engineer 74 zero-leakage features, including 58 causal historical features using prior-day network states.
3. Conduct comprehensive statistical testing with 14 formal hypothesis tests and daily cluster-robust ANOVA.
4. Select and evaluate the top four classification algorithms on the natural class distribution.
5. Select and evaluate the top four regression algorithms using robust L1 loss.
6. Deploy our complete methodology within an interactive, production-ready Streamlit platform."

---

### Slide 4: Dataset Description & Provenance
"Our dataset comes directly from the United States Department of Transportation and Bureau of Transportation Statistics TranStats database. It contains 3,000,000 raw records spanning 56 continuous months—from January 2019 to August 2023. 

The dataset captures the operations of 18 major US airlines, 378 commercial airports, and 7,785 unique directional flight routes. After cleaning, our analytical population consists of 2,920,859 non-cancelled commercial flights."

---

### Slide 5: Data Cleaning & Preprocessing Decisions
"Data cleaning was guided by strict domain realities. First, we identified and removed 72 duplicate records. Second, we excluded 79,140 cancelled flights because when a flight is cancelled, departure never takes place. However, we retained 8,261 diverted flights because they experienced a genuine departure pushback.

Most importantly, we preserved genuine extreme delays. The maximum delay in our dataset is 2,966 minutes—almost 50 hours—resulting from severe winter blizzards. In aviation data science, extreme delays are not measurement errors; they are critical operational realities. Furthermore, we rejected SMOTE or synthetic oversampling, strictly preserving the natural 82.3% on-time to 17.7% delayed operational ratio."

---

### Slide 6: Causal Feature Engineering Architecture
"How do we give our models predictive power without leaking future information? We engineered 74 pre-departure features.

In addition to scheduled departure hour, day of week, and distance, we developed 58 causal prior-day historical features. These calculate the delay rates and average delays of the airline, origin airport, and route over rolling 1-day, 7-day, 30-day, and cumulative historical windows. 

Crucially, every historical feature is computed strictly on dates *prior* to the target flight date ($t < \text{FL\_DATE}$). To prevent noise on low-frequency routes, we incorporated Empirical Bayes shrinkage, which automatically pulls sparse route estimates toward the global network average."

---

### Slide 7: Exploratory Data Analysis & Delay Dynamics
"Our exploratory analysis revealed striking operational dynamics. First, 64.9% of flights depart on time or early, with the most common departure occurring 5 minutes before schedule.

Second, delays accumulate monotonically throughout the day. Flights scheduled between 5:00 AM and 11:00 AM experience only a 10.8% delay rate, averaging 4.2 minutes of delay. But by the evening window, between 5:00 PM and 9:00 PM, the delay rate surges to 26.2%, averaging 17.8 minutes. This reflects the compounding effect of rotational aircraft delays as inbound planes arrive late throughout the day."

---

### Slide 8: Descriptive Statistics & The Non-Gaussian Nature of Delays
"Looking at our descriptive statistics, we see why standard statistical assumptions fail. The mean delay is +10.1 minutes, but the median is -2.0 minutes. This 12-minute gap between mean and median is a textbook indicator of extreme skewness.

In fact, our calculated skewness is +11.49, and excess kurtosis is +243.94. This confirms a heavily leptokurtic distribution with extreme right-tail density. Normality tests by Shapiro-Wilk and D'Agostino-Pearson conclusively reject Gaussian distribution with p-values below $10^{-85}$. This fundamental finding proved that traditional mean-squared-error regression would fail."

---

### Slide 9: Inferential Statistics: 14 Hypothesis Tests
"We conducted 14 formal two-tailed hypothesis tests, adjusting all p-values with the Benjamini-Hochberg False Discovery Rate procedure at $\alpha = 0.05$.

To highlight a few key findings: Test HT-04 proved the evening versus morning delay difference is statistically significant with a Cohen's d of 0.231. In Test HT-07, we performed a paired t-test across 1,704 matched calendar dates, controlling for daily weather shocks. The paired effect size was $d_z = 1.023$—a massive effect showing evening flights suffer an additional 13.6 minutes of delay on the same day. Finally, Test HT-11 showed that evening flights have 3.67 times higher odds of disruption compared to morning flights."

---

### Slide 10: ANOVA Suite & Multicollinearity (VIF)
"Because thousands of flights depart on the same calendar day, they share common weather and airport capacity shocks. This clustering can artificially inflate standard ANOVA F-statistics.

To address this, we implemented daily cluster-robust block-mean ANOVA. Even after aggregating flights into daily block means, carrier and day-part factors remain significant at $p < 10^{-15}$. 

We also performed Variance Inflation Factor analysis. We discovered severe multicollinearity between flight distance and scheduled block duration, with VIFs around 30. Removing scheduled duration collapsed distance VIF to 1.029, resolving collinearity."

---

### Slide 11: System Architecture & Chronological Validation
"Our system architecture enforces chronological integrity. We partitioned our 2.92 million flights by calendar date:
1. The Fit partition covers 2019 through 2021 with 1.78 million flights, used for training and 3-fold TimeSeriesSplit tuning.
2. The Validation partition covers early 2022 with 504,000 flights, used for model selection and threshold calibration.
3. The Out-of-Time Test partition covers late 2022 through August 2023 with 632,306 flights, kept completely untouched until final evaluation.

This guarantees that our models are tested strictly on future flights they have never seen."

---

### Slide 12: Classification Methodology & Algorithm Selection
"For classification, our objective is to predict whether a flight will experience a delay $\ge 15$ minutes.

Candidate algorithms were tuned and compared on our validation set. Based on validation delayed-class F1-score, we selected the top four architectures:
1. XGBoost Classifier, which achieved the highest validation F1 of 0.4510 at an optimal threshold $\tau^* = 0.2372$.
2. HistGradientBoosting Classifier.
3. LightGBM Classifier.
4. Calibrated Logistic Regression.

Notice our optimal decision threshold: by lowering the threshold from 0.50 to 0.237, we dramatically boosted disruption detection without incurring excessive false alarms."

---

### Slide 13: Classification Results & Champion Performance
"Here are the final test results on 632,306 untouched flights. 

Our champion, XGBoost Classifier, achieved an F1-score of 0.4522, an ROC-AUC of 0.7068, and a recall of 59.27% at an overall accuracy of 68.28%. Looking at the confusion matrix, out of 139,693 actual disruptions in the test set, our model successfully detected 82,801 flights before pushback. This provides real operational value for gate managers."

---

### Slide 14: Regression Methodology & Robust L1 Formulation
"Turning to regression, our objective is to estimate the exact delay in minutes.

Because delays have a skewness of 11.49, standard L2 loss models try to predict the conditional mean (+10 minutes), causing huge errors on the 65% of flights that depart on time. 
We formulated our regressors with L1 loss to target the conditional median (-2 minutes). We evaluated candidate models and selected the top four based on lowest validation MAE: LightGBM Regressor, HistGradientBoosting, XGBoost Regressor, and Gradient Boosting."

---

### Slide 15: Regression Results & Residual Diagnostics
"Evaluating our regression models on the 632,306 test flights, our champion LightGBM Regressor achieved an MAE of 18.56 minutes. This represents an 18.1% error reduction over the training mean baseline of 22.65 minutes.

Even more impressive is the Median Absolute Error of just 5.16 minutes. That means for more than half of all commercial flights, our model's prediction is accurate to within 5 minutes of actual departure. Our residual diagnostics show errors cleanly centered at zero, confirming robust median estimation."

---

### Slide 16: Model Comparison & Experiment Ledger
"This slide presents our consolidated benchmark. 

In classification, XGBoost leads with 0.7068 ROC-AUC and 0.4522 F1-score, with LightGBM and HistGradientBoosting performing closely behind. In regression, LightGBM leads with 18.56 minutes MAE and 5.16 minutes Median AE, followed by HistGradientBoosting at 18.60 minutes. All four active models for each task show strong consistency between validation and test partitions, proving zero overfitting."

---

### Slide 17: Interactive Streamlit Platform Demonstration
"All of our models and statistical analyses are deployed in an interactive 6-page Streamlit application.

The dashboard features executive KPI cards, an 8-tab statistical analysis engine, interactive confusion matrices, residual scatter plots, and live inference simulators. A dispatcher can select an airline, route, departure hour, and date, and the simulator automatically computes all 74 features to output a calibrated disruption probability and predicted delay duration in real time."

---

### Slide 18: Conclusion, Limitations & Future Scope
"To conclude: our project successfully established a zero-leakage, statistically validated system for flight delay intelligence across 2.92 million flights. By keeping the top four models for each task and eliminating synthetic sampling, we achieved operational realism with an ROC-AUC of 0.7068 and a median delay error of 5.16 minutes.

Our current limitation is that pre-departure timetables cannot observe sudden localized thunderstorms occurring minutes before pushback. In future work, we plan to integrate live NOAA METAR weather feeds and track physical aircraft tail rotations across the network. 

Thank you for your time and attention. We now welcome any questions."

