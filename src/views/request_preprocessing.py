


def change_types(data_dict):

    #first change gender to zero or one
    if data_dict['gender'] == 'male':
        data_dict['gender'] = 'M'
    else:
        data_dict['gender'] = 'F'

    
    # change age to range in 65-74 years    130764
    """ 
    55-64 
    75-84 
    25-34 
    35-44 
    45-54 
    18-24 
    85-89 
    ABOVE 90
    """
    if data_dict['age'] >= 55 and data_dict['age'] <= 64:
        data_dict['age'] = '55-64'
    elif data_dict['age'] >= 75 and data_dict['age'] <= 84:
        data_dict['age'] = '75-84'
    elif data_dict['age'] >= 25 and data_dict['age'] <= 34:
        data_dict['age'] = '25-34'
    elif data_dict['age'] >= 35 and data_dict['age'] <= 44:
        data_dict['age'] = '35-44'
    elif data_dict['age'] >= 45 and data_dict['age'] <= 54:
        data_dict['age'] = '45-54'
    elif data_dict['age'] >= 18 and data_dict['age'] <= 24:
        data_dict['age'] = '18-24'
    elif data_dict['age'] >= 85 and data_dict['age'] <= 89:
        data_dict['age'] = '85-89'
    elif data_dict['age'] >= 90:
        data_dict['age'] = 'ABOVE 90'
