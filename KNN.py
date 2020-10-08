def KNN_classifier(valid_x, valid_y, base_x, base_y, k=3):
    '''
    input x, y must be numpy arrays of this dataset, k is recommended positive odd number
    please import numpy as np
    '''
    assert len(base_x.shape) == len(valid_x.shape) == 2, "Expected 2D x input"
    assert len(base_y.shape) == len(valid_y.shape) == 1, "Expected 1D y target label"
    assert base_x.shape[0] == base_y.shape[0], "The row-counts between x and y for base dataset are not in line"
    assert valid_x.shape[0] == valid_y.shape[0], "The row-counts between x and y for validation dataset are not in line"
    assert base_x.shape[1] == valid_x.shape[1], "The col-counts mismatch for x"

    base_x_size = base_x.shape[0] 
    valid_x_size = valid_x.shape[0]
    x_feature_size = base_x.shape[1]
    
    '''
    map valid entry to base features
    '''
    valid_repeat_func = np.repeat(base_x_size,valid_x_size)
    valid_x_mat = np.repeat(valid_x, valid_repeat_func, axis=0).reshape(valid_x_size,-1,x_feature_size)
    base_x_mat = np.tile(base_x,[valid_x_size,1,1])

    '''
    elementwise difference
    '''
    diff_mat = valid_x_mat - base_x_mat

    '''
    calculate distance
    '''
    L2_mat = (((diff_mat)**2).sum(axis=2))**0.5

    '''
    sort and get k nearest neighbours
    '''
    sorted_L2 = np.argsort(L2_mat)
    sorted_L2_ind = np.argsort(L2_mat).transpose()
    label_array = np.array([])
    ref_k = int(k/2)

    for i in range(k):
        label_index = sorted_L2_ind[i]
        class_label = base_y[label_index]
        if len(label_array) == 0:
            label_array = class_label
        else:
            label_array = np.vstack([label_array,class_label])
    
    '''
    vote for prediction
    '''        
    temp_array = label_array.transpose()
    one_count = np.count_nonzero(temp_array == 1,axis=1)
    one_count_chk = one_count > ref_k

    '''
    prediction
    '''

    y_predict = one_count_chk.astype(int)
    
    return y_predict