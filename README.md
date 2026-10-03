\# Predicting Metabolic Cost During Human Walking



\## 1. Project Overview



This Machine Learning mini-project investigates the prediction of metabolic cost during human walking using biomechanical and physiological measurements.



Metabolic cost represents the energy expenditure associated with walking. Direct measurement of metabolic cost using laboratory equipment such as indirect calorimetry can be time-consuming and noisy. Therefore, this project investigates whether metabolic cost can be predicted from easier-to-obtain walking-related measurements.



The project is based on the research problem presented in:



\*\*Predicting Metabolic Cost During Human-in-the-Loop Optimization\*\*



by Erez Krimsky and Eley Ng.



For this mini-project, a publicly available biomechanics and energetics dataset is used to implement the related metabolic-cost prediction task.



\---



\## 2. Dataset



\### Dataset Name



\*\*A biomechanics and energetics dataset of neurotypical adults walking with and without kinematic constraints\*\*



The dataset contains walking measurements collected from neurotypical adults under different walking conditions.



The available measurements include:



\* Electromyography (EMG)

\* Ground reaction forces (GRF)

\* Joint-angle measurements

\* Metabolic measurements obtained using indirect calorimetry



\### Dataset Link



https://doi.org/10.26188/c.6887854.v1



The project uses the segmented MATLAB files provided with the dataset.



\---



\## 3. Problem Statement



The project treats metabolic-cost prediction as a supervised regression problem.



The input consists of biomechanical and physiological measurements obtained during walking, while the target is the measured metabolic cost.



The objective is to learn a relationship between the walking-related features and metabolic cost and evaluate how accurately the models can predict the target.



\---



\## 4. Features



The current feature set contains \*\*24 input features\*\*.



\### EMG Features



\* 16 mean EMG features



\### Ground Reaction Force Features



\* Peak vertical GRF — Left

\* Peak vertical GRF — Right



\### Joint Range-of-Motion Features



\* Hip ROM — Left

\* Hip ROM — Right

\* Knee ROM — Left

\* Knee ROM — Right

\* Ankle ROM — Left

\* Ankle ROM — Right



Therefore:



```text

16 EMG

\+ 2 GRF

\+ 6 Joint ROM

\----------------

24 Features

```



\---



\## 5. Target



The target variable is metabolic cost.



The target is calculated as the mean metabolic cost during the final 20% of each walking test.



The metabolic cost is represented in:



```text

W/kg

```



Each walking test is represented as one row in the generated feature dataset.



\---



\## 6. Feature Extraction



Feature extraction is implemented using:



```text

build\_features.py

```



The script processes the segmented subject data and extracts the features required for the machine-learning models.



For each walking test, the final 20% of the test is used for calculating the features and metabolic-cost target.



The generated feature files are stored in:



```text

features/

```



The combined dataset is:



```text

features/all\_subjects.csv

```



\---



\## 7. Dataset Organization



The segmented data for each subject is expected to follow this structure:



```text

Sub\_N/

├── SubN\_segEnergetics.mat

└── SubN\_segMechanics.mat

```



For example:



```text

Sub\_1/

├── Sub1\_segEnergetics.mat

└── Sub1\_segMechanics.mat

```



The large raw dataset files are not intended to be committed to the GitHub repository.



\---



\## 8. Repository Structure



The main project structure is:



```text

ml-metabolic-cost/

│

├── .gitignore

├── README.md

├── requirements.txt

│

├── build\_features.py

├── lasso\_baseline.py

├── neural\_net.py

├── ablation.py

├── demo.py

│

├── features/

│   ├── all\_subjects.csv

│   ├── sub1\_features.csv

│   ├── sub2\_features.csv

│   ├── sub3\_features.csv

│   ├── sub4\_features.csv

│   └── sub9\_features.csv

│

└── results/

```



\### Main Python Files



\*\*`build\_features.py`\*\*



Extracts the required features and target from the segmented dataset.



\*\*`lasso\_baseline.py`\*\*



Implements the LASSO regression baseline.



\*\*`neural\_net.py`\*\*



Implements the neural-network regression model.



\*\*`ablation.py`\*\*



Used to investigate the effect of different feature groups.



\*\*`demo.py`\*\*



Provides a demonstration of the project/model pipeline.



\---



\## 9. Setup



\### Step 1: Clone the Repository



```bash

git clone <REPOSITORY\_URL>

cd <REPOSITORY\_NAME>

```



\### Step 2: Create a Virtual Environment



On Windows:



```bash

python -m venv venv

```



Activate the environment:



```bash

venv\\Scripts\\activate

```



\### Step 3: Install Dependencies



```bash

pip install -r requirements.txt

```



\---



\## 10. Getting the Dataset



Download the required segmented dataset files from the dataset page:



https://doi.org/10.26188/c.6887854.v1



The required subject files are:



```text

SubN\_segEnergetics.mat

SubN\_segMechanics.mat

```



Place the files inside the corresponding subject directories.



For example:



```text

Sub\_1/

├── Sub1\_segEnergetics.mat

└── Sub1\_segMechanics.mat

```



Large raw dataset files should not be committed to GitHub.



\---



\## 11. Generating Features



After placing the required dataset files in the correct directories, run:



```bash

python build\_features.py

```



The script generates subject-level feature CSV files in:



```text

features/

```



and a combined dataset:



```text

features/all\_subjects.csv

```



\---



\## 12. Machine Learning Approach



The project uses regression models to predict metabolic cost.



The main models are:



1\. LASSO regression baseline

2\. One-hidden-layer neural network



The models are evaluated using cross-validation and Mean Squared Error (MSE).



The project also investigates the contribution of different feature groups through ablation analysis.



\---



\## 13. Evaluation Metric



The primary evaluation metric is:



\*\*Mean Squared Error (MSE)\*\*



MSE measures the average squared difference between the actual and predicted metabolic-cost values.



A smaller MSE indicates smaller prediction errors.



Cross-validation is used to evaluate model performance while reducing dependence on a single train-test split.



\---



\## 14. Relationship to the Original Research Paper



The original research paper studied metabolic-cost prediction using data collected during human-in-the-loop optimization with an assistive robotic device.



This mini-project uses a publicly available biomechanics and energetics dataset to implement the related prediction problem.



Therefore, the dataset and experimental setup used in this project are not identical to those used in the original paper.



The project focuses on the machine-learning problem of predicting metabolic cost from walking-related measurements.



\---



\## 15. Important Notes



\* The feature definitions and target calculation should remain consistent throughout the project.

\* Raw and large dataset files should not be committed to GitHub.

\* The generated CSV files contain the processed features used for machine-learning experiments.

\* The project uses supervised regression because the target is a continuous metabolic-cost value.



