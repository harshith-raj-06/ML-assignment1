# ML-assignment1

train_var1.py, train_var2.py is where all the degrees of lasso and ridge are compared (var1: degrees 1-10, var2: degrees 2-14). They pick the best alpha for each degree using 5-fold cross-validation and save the results and an MSE vs degree plot.

In var1_final.py, var2_final.py i take the best/good performing model from train and run them with some fine tuning of alpha, then train on the full training data and write the test predictions in the csv files.
