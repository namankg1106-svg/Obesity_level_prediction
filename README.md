# Obesity Level Prediction

A machine learning project that classifies people into seven obesity levels using their eating habits and physical condition. Built as a Data Science Minor Project (INT 324) at Lovely Professional University.

## Dataset
- **Source:** [UCI Machine Learning Repository: Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544)
- **Size:** 2,111 records, 17 columns (2,087 after removing 24 duplicates)
- **Population:** Individuals from Mexico, Peru and Colombia, aged 14 to 61
- **Target (`NObeyesdad`):** Insufficient Weight, Normal Weight, Overweight Level I and II, Obesity Type I, II and III
- **Note:** About 77% of the records are synthetic (generated with SMOTE) and 23% were collected from a web survey. Results should be read as a demonstration of method, not as a clinical tool.

## Approach
1. Load and clean the data (drop duplicates, check missing values)
2. Exploratory data analysis (class balance, age, categorical features, correlations, BMI)
3. Encode the target with LabelEncoder and one-hot encode categorical features (14 features become 29)
4. Stratified 80/20 train-test split (`random_state=42`) and StandardScaler
5. Train and compare four models, then evaluate with accuracy, macro precision, recall, F1, and confusion matrices
6. Random Forest feature importance
7. Leakage experiment: repeat with Height and Weight included

## Models and Results (habits only, Height and Weight excluded)
| Model | Accuracy | Macro F1 |
|---|---|---|
| **Random Forest** | **0.8397** | **0.8368** |
| KNN | 0.7608 | 0.7500 |
| Decision Tree | 0.6914 | 0.6879 |
| Logistic Regression | 0.6316 | 0.5981 |

**Top features:** Age, vegetable consumption (FCVC), number of main meals (NCP), physical activity (FAF), technology use (TUE), water intake (CH2O).

## Target Leakage Experiment
The label was derived from BMI (Weight / Height²). Setting `USE_HEIGHT_WEIGHT = True` raises accuracy to about 93% for the Decision Tree (0.9354) and Random Forest (0.9306), because the models simply re-learn the BMI formula. The honest result is the habits-only run above.

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn
python obesity_level_prediction.py
```
Keep `ObesityDataSet.csv` in the same folder as the script. Change `USE_HEIGHT_WEIGHT` at the top of the script to switch between the two experiments.

## Repository Contents
- `obesity_level_prediction.py`: full analysis script
- `ObesityDataSet.csv`: dataset
- `Obesity_Level_Prediction_Project_Report.docx`: full project report

## Tech Stack
Python 3.12, pandas, NumPy, scikit-learn, Matplotlib

## Limitations
- A large share of the data is synthetic, so test scores may be optimistic
- Results come from a single train-test split, with no cross-validation or hyper-parameter tuning
- The sample is young adults from three countries and does not generalise to other groups

## Author
**Naman Gupta** (Reg. No. 12406422, Section K3P24MW)
Supervisor: Dr. Madhu Bala, Lovely Professional University

## Citation
Palechor, F. M., and De la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. *Data in Brief*, 25, 104344. https://doi.org/10.1016/j.dib.2019.104344
