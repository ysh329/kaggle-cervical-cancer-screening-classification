import os, urllib
import mxnet as mx
import cv2 
import numpy as np
import sys

def download(url,prefix=''):
    filename = prefix+url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)

def get_image(url, show=False):
    filename = url.split("/")[-1]
    urllib.urlretrieve(url, filename)
    img = cv2.imread(filename)
    if img is None:
        print('failed to download ' + url)
    if show:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    return filename

def predict(filename, mod, synsets, Batch, resize_shape=(224, 224)):
    # Oops!!! cv2 crashed!
    #img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    from skimage import io
    img = io.imread(filename)
    if img is None:
        return None
    #img = cv2.resize(img, resize_shape)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    img = img[np.newaxis, :] 
    
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    #print ",".join(['0.jpg'] + map(str, prob))

    #a = np.argsort(prob)[::-1]    
    #for i in a[0:5]:
        #print('probability=%f, class=%s' %(prob[i], synsets[i]))
    return prob

def main():

    # inialization paramter
    #test_dir = '/home/yuanshuai/data/ccs/test_seg_224'
    test_dir = "/home/yuanshuai/data/ccs/test_stg2_seg_224"
    resize_shape = (224, 224)
    batch_size = 1
    num_gpus = 1
    data_shape = resize_shape
    #model_prefix = "./models/resnet-50-train-seg-224/resnet-50-train-seg-224"
    print sys.argv, len(sys.argv)
    model_prefix = sys.argv[1] #"./inception-resnet-v2-50-train-seg-224-lr-0.01/inception-resnet-v2-50-train-add-seg-224-lr-0.01"
    epoch = int(sys.argv[2]) # sys.argv[0] is this python-file-self name
    csv_name = "".join([ model_prefix[:model_prefix.index("/", 2)+1], "[result]", model_prefix.split('/')[-1], "-"+str(epoch)] )+".csv"

    with open('full-synset.txt', 'r') as f:
        synsets = [l.rstrip() for l in f]

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)


    # Create a model for this model on GPU 0.
    #devs = [mx.gpu(i) for i in xrange(num_gpus)]
    devs = [mx.gpu(2)]
    mod = mx.mod.Module(symbol=sym, context=devs)
    #mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3,data_shape[0],data_shape[1]))])
    mod.set_params(arg_params, aux_params)

    # Next we define the function to obtain an image by a given URL and the function for predicting.

    import matplotlib
    matplotlib.rc("savefig", dpi=100)
    import matplotlib.pyplot as plt
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])


    # We are able to classify an image and output the top predicted classes.
    #url = 'http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg'
    #predict(get_image(url), mod, synsets)
    #predict('./Cat-hd-wallpapers.jpg', mod, synsets, Batch)

    
    # make prediction for a collection of images
    #test_dir = '/home/yuanshuai/data/ccs/test'
    test_img_dir_list = map(lambda test_img_name: "/".join([test_dir, test_img_name]), os.listdir(test_dir))
    test_img_prob_list = []

    import csv
    csvfile = open(csv_name,'wb')
    csvfile.write(",".join(['image_name', 'Type_1', 'Type_2', 'Type_3\n']))

    for test_img_idx in xrange(len(test_img_dir_list)):#xrange(3):
        #test_img_dir = test_img_dir_list[test_img_idx]
        test_img_dir = test_img_dir_list[test_img_idx]#test_dir+'/'+str(test_img_idx)+".jpg"
        print test_img_dir
        test_img_prob = predict(test_img_dir, mod, synsets, Batch)

        row = ",".join([test_img_dir.split('/')[-1]] + map(str, test_img_prob))+'\n'
        #print row
        csvfile.write(row)

    csvfile.close()


if __name__ == "__main__":
    main()
