
def get_params(theta_d):
    if theta_d <= 100:
        params = {"maxIter": 2001,  "nSamples": 500, 
                "lr": 4e-4, "lr_prec": [8e-2,5e-2]}
    if theta_d == 50:
        params = {"maxIter": 2001,  "nSamples": 500, 
                "lr": 4e-3, "lr_prec": [8e-2,5e-2]}
    if theta_d == 100:
        params = {"maxIter": 3001,  "nSamples": 400, 
                "lr": 4e-3, "lr_prec": [8e-2,5e-2]}
    if theta_d == 500:
        params = {"maxIter": 5001,  "nSamples": 400, 
                "lr": 1e-2, "lr_prec": [7e-2,3e-2]}
    if theta_d == 1000:
        params = {"maxIter": 6001,  "nSamples": 400, 
                "lr": 1e-2, "lr_prec": [3e-2, 1e-2]}
    return params