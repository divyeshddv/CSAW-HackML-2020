import keras
import sys
import h5py
import numpy as np

clean_data_filename = str(sys.argv[1])
backdoored_data_filename = str(sys.argv[2])
model_filename = str(sys.argv[3])

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def data_getsample(x, y, n):
    i=0
    sampledx = []
    sampledy = []
    while len(sampledy)<n:
        if y[i] not in sampledy:
            sampledx.append(x[i])
            sampledy.append(y[i])
        i+=1
    return np.array(sampledx), np.array(sampledy)
            


def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    n = max(y_test)
    x_cleansample, y_cleansample = data_getsample(x_test, y_test, 10)
    
    x_testb, y_testb = data_loader(backdoored_data_filename)
    x_testb = data_preprocess(x_testb)
    
    bd_model = keras.models.load_model(model_filename)
    label_p = []
    print(np.array([x_testb[0]]).shape)
    print(y_testb.shape)
    
    for i in range(len(x_testb)):
        if i%1000==0:
            print(i)
        x_added = []
        for j in range(len(x_cleansample)):
            x_added.append((x_cleansample[j]+x_testb[i])/2)
        x_added = np.array(x_added)
        preds = np.argmax(bd_model.predict(x_added), axis=1)
        if len(np.unique(preds)) < len(preds)/2:
            label_p.append(n+1)
        else:
            label_p.append(np.argmax(bd_model.predict(np.array([x_testb[i]])), axis=1)[0])
            
    label_p = np.array(label_p)
    class_accu = np.mean(np.equal(label_p, y_testb))*100

    label_b = np.argmax(bd_model.predict(x_testb), axis=1)
    class_accub = np.mean(np.equal(label_b, y_testb))*100
    
    print('Network Classification accuracy on dataset: ', backdoored_data_filename ," = ", class_accub)
    print('Repaired Network Classification accuracy on dataset: ', backdoored_data_filename ," = ", class_accu)
    
if __name__ == '__main__':
    main()
