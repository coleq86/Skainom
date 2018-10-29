# load required libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
from keras.applications.inception_v3 import preprocess_input
from preprocessing import preprocessing

def predict_status(img_height, img_width, n_images, path_test, preprocess=True):
    # dimensions of our images.
    img_width, img_height = img_width, img_height

    test_data_dir = path_test

    # if preprocessing needs to be done - preprocessed images are stored in preprocessed folder
    if preprocess == True:
        preprocessing(img_width,img_height, test_data_dir)

    nb_test_samples = n_images

    # number of images in each batch
    batch_size = 10

    # this is the augmentation configuration we will use for testing:
    # only rescaling & default preprocessing function for InceptionV3
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    # load saved model for predictions
    model = load_model('Best_Model_Checkpoint.hdf5')

    # generate predictions
    pred = model.predict_generator(test_generator, steps=nb_test_samples / batch_size)

    # with 0.5 as threshold get Normal/ abnormal class
    y_pred = (pred >= 0.5).astype(int)

    # store predictions along with labels in dataframe
    predictions = pd.DataFrame()
    predictions['filename'] = test_generator.filenames
    predictions['pred'] = y_pred

    predictions.ix[predictions['pred'] == 1, 'pred'] = 'Normal'
    predictions.ix[predictions['pred'] == 0, 'pred'] = 'Abnormal'

    predictions.to_csv('prediction_file_testset.csv', index=False)

if __name__ == "__main__":
    path = "data/valid"
    predict_status(300, 300, 200, path, True)