import sys
print(sys.version)
try:
    from flaml import AutoML
    model = AutoML()
    import numpy as np
    X_train = np.array([[1,2],[3,4]])
    y_train = np.array([0, 1])
    model.fit(X_train, y_train, task="classification", time_budget=1)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
