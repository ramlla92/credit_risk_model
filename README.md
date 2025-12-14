## Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord is a set of international rules that guide how banks manage and measure credit risk. Its main goal is to ensure that banks remain financially stable by carefully assessing the risk of lending money to customers. Under Basel II, banks are not only required to measure credit risk accurately, but also to clearly explain how that risk is calculated and how lending decisions are made.

This means that banks must use credit scoring models that humans can understand. If a customer is approved or rejected for a loan, the bank should be able to explain the decision using clear reasons, such as the customer’s transaction behavior or spending patterns. Regulators and internal risk teams should also be able to review the model logic and verify that it is fair, consistent, and reasonable.

Because of this, credit risk models must be transparent, interpretable, and well-documented. Model explainability is not just a technical preference, but a regulatory requirement. A clear and explainable model helps banks justify decisions, pass regulatory audits, and build trust with customers and stakeholders.

### 2. Proxy Default Variable and Associated Business Risks

In this project, there is no direct information showing whether a customer has failed to repay a loan, because the buy-now-pay-later service is new. However, machine learning models need a clear target variable to learn what makes a customer risky or safe. To solve this problem, we create a proxy default variable, which acts as a substitute for real default data.

This proxy is created using customer behavior, specifically Recency, Frequency, and Monetary (RFM) metrics. These metrics describe how recently a customer made a transaction, how often they transact, and how much money they spend. Customers who transact very rarely, have low spending amounts, and have not been active for a long time are considered less engaged. Such customers are labeled as high risk because low engagement can indicate a higher chance of not repaying credit in the future.

Using a proxy variable allows the model to be trained, but it also introduces business risk. Since the proxy does not represent actual loan defaults, some customers may be misclassified. The model may wrongly reject good customers (false positives) or approve customers who later fail to repay (false negatives). For this reason, predictions based on proxy labels must be used carefully and should be continuously reviewed and validated once real repayment data becomes available.

### 3. Trade-offs Between Interpretable and Complex Models

Interpretable models, such as Logistic Regression with Weight of Evidence (WoE), are commonly used in credit risk modeling because their decisions are easy to understand and explain. These models clearly show how each feature, such as transaction frequency or spending behavior, affects a customer’s risk level. This transparency makes them suitable for regulated environments, where models must be reviewed, justified, and audited by regulators and internal risk teams.

More complex models, such as Gradient Boosting, can achieve better predictive performance by learning complex and non-linear patterns in the data. However, these models are harder to interpret and explain. This lack of transparency can make regulatory approval, model validation, and stakeholder trust more challenging.

The main trade-off is between predictive accuracy and explainability. In regulated financial settings, a common best practice is to use interpretable models as the primary decision-making tool, while using complex models as benchmarks to evaluate performance improvements. This approach balances regulatory compliance with strong predictive capability.