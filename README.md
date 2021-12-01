We hypothesized that clinical features-based machine learning algorithms could predict influenza infection under ED and primary care settings among patients with influenza-like illness (ILI).
We found XGBoost performed the best among the seven algorithms with accuracy and AUC and outperformed conventional models.

The AUROC for training and testing sets trained by XGBoost algorithm
![ROC curve for XGBoost](https://user-images.githubusercontent.com/55526809/144149402-92a615e2-7df8-4ec8-b847-b8436b8f4b59.png)


To explain how the machine learning model made the prediction, we used the Shapley Additive exPlanations (SHAP) value to evaluate the output of the models that help to understand the direction and strength of the selected features in the final model.
![shape](https://user-images.githubusercontent.com/55526809/144149687-f40fe979-3a64-4521-974d-dba4da21f21e.png)


The calibration curve for goodness-of-fit diagnosis for XGBoost algorithm
![calibration plot for XGBoost](https://user-images.githubusercontent.com/55526809/144149781-295ca60c-6e6a-4a32-95ac-95971cf2c55f.png)
