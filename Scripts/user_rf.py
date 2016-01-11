# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, n_jobs=20, random_state=0, verbose=1)
rf.fit(X_train, y_train)

preds = rf.predict_proba(X_val)
ndcg5(preds, xg_val)

y_pred_rf = rf.predict_proba(X_test)
