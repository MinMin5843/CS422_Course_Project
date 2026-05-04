# Computer Science Outcomes & Job Posting Competitveness
This a course project for Introduction to Machine Learning (CS422) of the Spring semester of 2026.
## Project Overview
This project has two main tasks:

* **Task 1:** Student Career Path Prediction (The files for this task include the task1_preprocess.py and task1_model.py files.)
    * *Goal:* Predict a computer science student's future career based on the following:
        - Demographics (Gender, Age)
        - Academic performance (GPA)
        - Major
        - Interested Domain
        - Projects completed
        - Skill levels in Python, SQL, and Java (from Weak to Moderate to Strong)
    * *Dataset Used:* Patil, R. (2024). *Computer Science Students Career Prediction*. Kaggle. https://www.kaggle.com/datasets/devildyno/computer-science-students-career-prediction
    * *Target Variable:* Future Career (this is a multi-class classification provided within the dataset)
    * The Approach:
        1. Preprocessing (this is seen in the task1_preprocess.py file)
            - Ordinal‑encode skill levels:
                - Weak → 1  
                - Moderate → 2  
                - Strong → 3  
            - One‑hot encode categorical features:
                - Gender  
                - Major  
                - Interested Domain  
                - Projects  
            - Train/test split (80/20, stratified)
        2. Models evaluated (this is seen in the task1_model.py file):
            - Logistic Regression  
            - Random Forest  
            - Gradient Boosting
    * Output for this task: The files associated with this task will output accuracy and full classification report for each model. 

* **Task 2:** Job Posting Competitiveness Prediction (The files for this task include the task2_preprocess.py and task2_model.py files.) 
    * *Goal:* Predict how competitive a job posting is using job attributes to include the:
        - Salary range
        - Required skills
        - Experience required
        - Company size
        - Job category
        - Education level
    * *Dataset User:* Jawad, M. (Mar. 2026). *Tech Jobs, Salaries, and Skills (Version 1)*. Kaggle. https://www.kaggle.com/datasets/mjawad17/tech-jobs-salaries-and-skills-dataset
    * *Target Variable:* The difficulty category, otherwise shown as 'difficulty_cat' with labels of Low, Medium, or High.
        * The difficulty score is calculated using the following before being placed into three categories:
            - Number of required skills
            - Experience required
            - Average salary
            - Company size (larger companies tend to have a harder and longer interviewing process)
    * The Approach: 
        1. Preprocessing (this is seen in the task2_preprocess.py file)
            - Extract the number of skills from the comma‑separated list  
            - Compute the average salary  
            - Map the company size to a difficulty weight  
            - Compute the difficulty score  
            - One‑hot encode categorical features  
            - Train/test split (80/20, stratified)
        2. Models evaluated (this is seen in the task2_model.py file):
            - Logistic Regression  
            - Random Forest  
            - Gradient Boosting
    * Output for this task: The files associated with this task will output accuracy and full classification report for each model. 
* **Connecting both tasks:** To relate student interests to job market difficulty the following was done in the combo_analysis.ipynb file:
    1. Each student's Interested Domain was mapped to a Job Category.
        - Example: 
            - Software Engineering → Software Development
            - Artificial Intelligence → Technology  
    2. The job postings in those categories were then analyzed to determine the average difficulty, skill expectations, and salary competitiveness. 
* **For the other notebooks:**
    * The task1_eda.ipynb file explores and analyzes the Computer Science Students Career Prediction dataset.
    * The task2_eda.ipynb file explores and analyzes the Tech Jobs, Salaries, and Skills (Version 1) dataset. 

## How to run the program:

1. Create and activate a virtual environment.
2. Install the dependencies with the following command: 

    ```bash 
    pip install -r requirements.txt
3. Run Task 1 with the following command:
    
    ```bash
    python main_task1.py
4. Run Task 2 with the following command:
    
    ```bash
    python main_task2.py
