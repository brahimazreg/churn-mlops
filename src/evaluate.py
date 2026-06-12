from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)


def evaluate_model(model,
    X_test,
    Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(Y_test, proba)

    report = classification_report(Y_test, Y_pred)
    

    return  accuracy , roc_auc , report 

