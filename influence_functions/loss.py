def del_spd_del_theta(model, X_test_orig, X_test, dataset):
    num_params = len(convert_grad_to_ndarray(list(model.parameters())))
    del_f_protected = np.zeros((num_params,))
    del_f_privileged = np.zeros((num_params,))
    if dataset == 'german':
        numPrivileged = X_test_orig['age'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'compas':
        numPrivileged = X_test_orig['race'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'adult':
        numPrivileged = X_test_orig['gender'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'traffic':
        numPrivileged = X_test_orig['race'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'sqf':
        numPrivileged = X_test_orig['race'].sum()
        numProtected = len(X_test_orig) - numPrivileged
    elif dataset == 'random':
        numPrivileged = X_test_orig['AA'].sum()
        numProtected = len(X_test_orig) - numPrivileged

    for i in range(len(X_test)):
        del_f_i = del_f_del_theta_i(model, X_test[i])
        del_f_i_arr = convert_grad_to_ndarray(del_f_i)
        if dataset == 'german':
            if X_test_orig.iloc[i]['age'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['age'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'compas':
            if X_test_orig.iloc[i]['race'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['race'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'adult':
            if X_test_orig.iloc[i]['gender'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['gender'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'traffic':
            if X_test_orig.iloc[i]['race'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['race'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'sqf':
            if X_test_orig.iloc[i]['race'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['race'] == 0:
                del_f_protected += del_f_i_arr
        elif dataset == 'random':
            if X_test_orig.iloc[i]['AA'] == 1: #privileged
                del_f_privileged += del_f_i_arr
            elif X_test_orig.iloc[i]['AA'] == 0:
                del_f_protected += del_f_i_arr

    del_f_privileged /= numPrivileged
    del_f_protected /= numProtected
    v = del_f_protected - del_f_privileged
    return v

def del_f_del_theta_i(model, x, retain_graph=False):
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(model(torch.Tensor(x)), w, retain_graph=retain_graph)

def get_del_F_del_theta(model, X_test_orig, X_test, y_test, dataset, metric):
    if metric == 0:
        v1 = del_spd_del_theta(model, X_test_orig, X_test, dataset)
    elif metric == 1:
        v1 = del_tpr_parity_del_theta(model, X_test_orig, X_test, y_test, dataset)
    elif metric == 2:
        v1 = del_predictive_parity_del_theta(model, X_test_orig, X_test, y_test, dataset)
    else:
        raise NotImplementedError
    return v1

def convert_grad_to_ndarray(grad):
    grad_list = list(grad)
    grad_arr = None
    for i in range(len(grad_list)):
        next_params = grad_list[i]

        if isinstance(next_params, torch.Tensor):
            next_params = next_params.detach().squeeze().numpy()

        if len(next_params.shape) == 0:
            next_params = np.expand_dims(next_params, axis=0)

        if len(next_params.shape) > 1:
            next_params = convert_grad_to_ndarray(next_params)

        if grad_arr is None:
            grad_arr = next_params
        else:
            grad_arr = np.concatenate([grad_arr, next_params])

    return grad_arr