import pickle

path = r'D:\saved_model\trained_model.sav'

with open(path, 'rb') as file:
    model = pickle.load(file)

print(model.train)
