# Featured Projects

## Project: Customer LTV & Retention Analytics
- **Category**: Customer Analytics / Business Analytics / Machine Learning
- **Tech Stack**: Python, SQL, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn, SQLite, Git, GitHub
- **GitHub Repository**: https://github.com/Prakhar1709/Customer-Ltv-and-user-acquisition

### Overview & Business Problem
An end-to-end customer analytics and predictive modeling project that investigates customer lifetime value (LTV), purchasing behavior, retention patterns, and repeat-purchase opportunities for an e-commerce business using the Olist Brazilian E-Commerce dataset (99K orders, 93,358 unique customers, $15.42M total revenue).

### Key Methodologies & Workflow
- **RFM Segmentation**: Segmented 93,358 customers into Champions (8,030 customers / 17.39% revenue), Loyal Customers (31,507 / 43.89% revenue), Potential Loyalists (38,393 / 32.22% revenue), At Risk (12,267 / 5.52% revenue), and Low Value (3,161 / 0.98% revenue).
- **Cohort Retention Analysis**: Tracked monthly customer retention, revenue by acquisition cohort, and cumulative customer value over time.
- **Repeat Purchase Gap**: Revealed that only ~3% (2,801) of customers make repeat purchases, accounting for 5.6% of revenue, identifying a critical retention optimization opportunity.
- **Feature Engineering**: Extracted early behavioral signals (`early_orders`, `early_revenue`, `early_items`, `days_active`, `avg_order_value`, `orders_per_active_day`, `revenue_per_item`).
- **Predictive Modeling**: Formulated 90-day repeat purchase prediction (`repeat_purchase_90d`) using XGBoost. Handled 76.27:1 positive-to-negative class imbalance using `scale_pos_weight` and evaluated with PR-AUC (0.6068).
- **Chronological Split**: Enforced realistic time-based train/test splitting (69,539 train / 17,385 test) to prevent data leakage.
- **Threshold Optimization**: Swept probability thresholds to select `0.85`, targeting a high-efficiency group of 117 customers that captured 58.21% of repeat purchasers and 81.73% ($40,041.07) of repeat customer revenue.
- **Interactive Dashboard**: Deployed an interactive Streamlit application with modules for Customer KPIs, RFM segmentation, Cohort retention, and ML targeting.

---

## Project: Credit Card Fraud Analytics & Detection
- **Category**: Data Analytics / Business Analytics / Machine Learning
- **Tech Stack**: Python, SQL, Power BI, Scikit-learn, Logistic Regression, Random Forest, SMOTE, GridSearchCV, Pandas, NumPy, Matplotlib, Seaborn
- **GitHub Repository**: https://github.com/Prakhar1709/Credit-card-fraud-analytics-and-detection

### Overview & Business Problem
An end-to-end fraud analytics and classification system analyzing 284,807 transactions (492 fraudulent transactions, 0.17% fraud rate, $60.13K fraud amount) to detect illicit transaction patterns, highlight high-risk segments, and predict fraudulent activity.

### Key Methodologies & Workflow
- **Risk Segmentation & SQL Layer**: Transformed transaction-level data in SQL across amount buckets and hourly intervals. Discovered that transactions exceeding $500 exhibited the highest fraud rate (~0.369%), and identified significant time-of-day fraud spikes.
- **Power BI Monitoring Dashboard**: Built interactive KPI cards (Total Transactions, Fraud Count, Fraud Rate, Total Fraud Amount) with amount bucket distributions and hourly risk monitoring charts.
- **Imbalanced Learning with SMOTE**: Solved severe 0.17% class imbalance using Synthetic Minority Over-sampling Technique (SMOTE) paired with Logistic Regression and Random Forest Classifiers.
- **Hyperparameter Tuning & Threshold Tuning**: Applied GridSearchCV for parameter optimization and evaluated precision-recall trade-offs to minimize costly false negatives in high-value fraud detection.

---

## Project: Student Performance Indicator
- **Category**: Machine Learning / End-to-End ML Engineering
- **Tech Stack**: Python, Scikit-learn, CatBoost, XGBoost, Flask, Pandas, NumPy, Matplotlib, Seaborn, Dill, HTML/CSS, Git, GitHub
- **GitHub Repository**: https://github.com/Prakhar1709/Student-performance-indicator

### Overview & Business Problem
A production-grade, end-to-end Machine Learning web application that predicts a student's mathematics score based on demographic (gender, race/ethnicity, parental education, lunch type, test preparation) and academic factors (reading and writing scores).

### Key Methodologies & Workflow
- **Complete ML Engineering Lifecycle**: Structured pipeline covering data ingestion, data validation, exploratory data analysis, transformation, multi-model evaluation, and web deployment.
- **Model Comparison & Selection**: Evaluated and benchmarked multiple regression algorithms including Scikit-learn models, CatBoost, and XGBoost to select the top-performing predictor.
- **Modular Pipeline Architecture**: Built reusable preprocessing and prediction pipelines with artifact persistence (`dill`), ensuring consistent transformation between training and real-time user input.
- **Centralized Logging & Custom Exceptions**: Implemented structured logging and custom exception handlers across the application for complete execution traceability and debugging.
- **Flask Web Deployment**: Deployed the trained inference model into a lightweight, responsive Flask web application for interactive real-time score estimation.
