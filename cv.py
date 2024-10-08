import pandas as pd

#unseen drug
def cv2(data:pd.DataFrame):
    data_size = len(data)
    train, test = data[:int(data_size * 0.8)], data[int(data_size * 0.8):]

    train_data = dict(SMILES=[], Protein=[], Y=[])

    test_data = dict(SMILES=[], Protein=[], Y=[])
    protein_list = train["Protein"].drop_duplicates()

    for i in test.values:
        if i[1] in protein_list.values:
            test_data['SMILES'].append(i[0])
            test_data['Protein'].append(i[1])
            test_data['Y'].append((i[2]))

    test = pd.DataFrame(test_data)
    test_drug = test['SMILES'].drop_duplicates()
    test_size = len(test)

    for i in train.values:
        if i[0] not in test_drug.values:
            train_data['SMILES'].append(i[0])
            train_data['Protein'].append(i[1])
            train_data['Y'].append((i[2]))

    train = pd.DataFrame(train_data)
    val, test = test[:int(test_size * 1/3)], test[int(test_size * 1/3):]

    return train, val, test

#unseen protein
def cv3(data:pd.DataFrame):
    data_size = len(data)
    train, test = data[:int(data_size * 0.8)], data[int(data_size * 0.8):]

    train_data = dict(SMILES=[], Protein=[], Y=[])
    test_data = dict(SMILES=[], Protein=[], Y=[])
    drug_list = train["SMILES"].drop_duplicates()

    for i in test.values:
        if i[0] in drug_list.values:
            test_data['SMILES'].append(i[0])
            test_data['Protein'].append(i[1])
            test_data['Y'].append((i[2]))

    test = pd.DataFrame(test_data)
    test_protein = test['Protein'].drop_duplicates()
    test_size = len(test)

    for i in train.values:
        if i[1] not in test_protein.values:
            train_data['SMILES'].append(i[0])
            train_data['Protein'].append(i[1])
            train_data['Y'].append((i[2]))

    train = pd.DataFrame(train_data)
    val, test = test[:int(test_size * 1/3)], test[int(test_size * 1/3):]

    return train, val, test

#unseen both
def cv4(data:pd.DataFrame):
    data_size = len(data)
    train, test = data[:int(data_size * 0.8)], data[int(data_size * 0.8):]

    test_drug = test['SMILES'].drop_duplicates()
    test_protein = test['Protein'].drop_duplicates()
    train_data = dict(SMILES=[], Protein=[], Y=[])
    test_size = len(test)
    for i in train.values:

        if i[0] not in test_drug.values and i[1] not in test_protein.values:
            train_data['SMILES'].append(i[0])
            train_data['Protein'].append(i[1])
            train_data['Y'].append((i[2]))

    train = pd.DataFrame(train_data)
    val, test = test[:int(test_size * 1/3)], test[int(test_size * 1/3):]

    return train, val, test