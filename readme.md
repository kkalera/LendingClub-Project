# LendingClub Machine Learning

&nbsp;


**Context**

I've been asked to work on this project, approaching it from the following angle:


Imagine that you are a data scientist who was just hired by the LendingClub.
They want to automate their lending decisions fully, and they hired you to lead this project.
Your team consists of a product manager to help you understand the business domain and a software engineer who will help you integrate your solution into their product.
During the initial investigations, you've found that there was a similar initiative in the past, and luckily for you, they have left a somewhat clean dataset of LendingClub's loan data.
The dataset is located in a public bucket here: <https://storage.googleapis.com/335-lending-club/lending-club.zip> (although you were wondering if having your client data in a public bucket is such a good idea).
In the first meeting with your team, you all have decided to use this dataset because it will allow you to skip months of work of building a dataset from scratch.
In addition, you have decided to tackle this problem iteratively so that you can get test your hypothesis that you can automate these decisions and get actual feedback from the users as soon as possible.
For that, you have proposed a three-step plan on how to approach this problem.
The first step of your plan is to create a machine learning model to classify loans into accepted/rejected so that you can start learning if you have enough data to solve this simple problem adequately.
The second step is to predict the grade for the loan, and the third step is to predict the subgrade and the interest rate.
Your team likes the plan, especially because after every step, you'll have a fully-working deployed model that your company can use.
Excitedly you get to work!

&nbsp;

The project is divided into 3 parts:

- **Part 1:** Loan Approval Classification (this part)

- **Part 2:** Loan Grade Classification

- **Part 3:** Loan Sub Grade Classification And Interest Rate Regression


&nbsp;

**Project Structure**
```
.
└── Project home
    ├── src
    │   ├── data
    │   │   ├── p1_sample.json
    │   │   ├── p2_sample.json
    │   │   ├── p3_int_sample.json
    │   │   └── p3_sub_sample.json
    │   ├── lib
    │   │   ├── deployment.py
    │   │   └── functions.py
    │   ├── logs
    │   │   └── my.log
    │   └── requirements.txt
    ├── 335.ipynb
    ├── Project Part 1.ipynb
    ├── Project Part 2.ipynb
    ├── Project Part 3.ipynb
    └── readme.md
```

&nbsp;

**Usage**

If you'd like to run the notebook yourself, you can download the data using [this link](https://storage.googleapis.com/335-lending-club/lending-club.zip) and extract the two csv files into the `src/data/` folder and make sure they are named like:

- accepted_2007_to_2018Q4.csv
- rejected_2007_to_2018Q4.csv

Keep in mind that these datasets are relatively large and might take up considerable resources on your machine if you're running the notebooks.