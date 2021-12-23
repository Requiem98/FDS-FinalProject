# sensitivity analysis of k in k-fold cross-validation
from library import *
from dataset_creation import *


# evaluate the model using a given test condition
def evaluate_model(cv, X, y, model):
    
	# get the model
    
    if(model == "logistic"):
        model = LogisticRegression(solver="newton-cg")
    elif(model == "kneighbours"):
        pass
    else:
        raise(KeyError("Model not specifiedS"))
    
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()

def k_fold_choose(X, y, model = "logistic"):
    # calculate the ideal test condition
    ideal, _, _ = evaluate_model(LeaveOneOut(), X_res, y_res, model)

    # define folds to test
    folds = range(2,15)
    # record mean and min/max of each set of results
    means, mins, maxs = list(),list(),list()
    # evaluate each k value
    for k in tqdm(folds):
        # define the test condition
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
        # evaluate k value
        k_mean, k_min, k_max = evaluate_model(cv, X, y, model)
        # store mean accuracy
        means.append(k_mean)
        # store min and max relative to the mean
        mins.append(k_mean - k_min)
        maxs.append(k_max - k_mean)

    fig, ax = plt.subplots(1,1, figsize=(25,15))
    ax.grid(color="w")
    # line plot of k mean values with min/max error bars
    ax.errorbar(folds, means, yerr=[mins, maxs], fmt='o', color="k", ecolor="orange", elinewidth=3)
    # plot the ideal case in a separate color
    ax.plot(folds, [ideal for _ in range(len(folds))], color='r', linewidth=3.0)

    plt.xlabel("number of folds k", fontsize=50)
    plt.ylabel("Accuracy", fontsize=50)
    plt.xticks(size = 25)
    plt.yticks(size = 25)
    # show the plot
    fig.savefig("immages/errorbar.png", bbox_inches='tight')