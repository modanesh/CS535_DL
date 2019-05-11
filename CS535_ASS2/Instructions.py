# Loading the data
# Now, the data should contain these items:
# dict["train_data"]: a 10000 x 3072 matrix with each row being a training image (you can visualize the image by reshaping the row to 32x32x3
# dict["train_labels"]: a 10000 x 1 vector with each row being the label of one training image, label 0 is an airplane, label 1 is a ship.
# dict["test_data"]: a 2000 x 3072 matrix with each row being a testing image
# dict["test_labels]: a 2000 x 1 vector with each row being the label of one testing image, corresponding to test_data.
import cPickle

dict = cPickle.load(open("cifar_2class_py2.p","rb"))

for i in dict:
    print i, dict[i].shape