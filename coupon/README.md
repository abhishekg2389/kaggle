This repo contains code for the [Recruit Ponpare Coupon Prediction Challenge](https://www.kaggle.com/c/coupon-purchase-prediction/). It gives map@10 score 0.0072 on Private Leaderboard.

**Platform Used:** r3.8xlarge ami - AWS

**Rquirement:**

1. R 3.2.2
2. xgboost
3. Mono 2.8.x or another recent .NET runtime (Mono 2.10.x recommended)
4. Packages in RUNME.R

**To run do the following:**

1. [Download the data](https://www.kaggle.com/c/coupon-purchase-prediction/data)
2. Download the repo at the same location.
3. Create a new folder 'raw_data' and unzip the data files in the folder.
4. Install packages as required in RUNME.R
5. Modify paths and run RUNME.R
6. A submission folder is then generated containing the submissions.
 

Details about the implementation can be found in this blog post:
https://abhishekg2389.wordpress.com/2015/12/18/coupon-purchase-prediction/
