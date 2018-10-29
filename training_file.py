from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
import numpy as np
from keras.layers import BatchNormalization
from PIL import ImageFile
from preprocessing import preprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True

# dimensions of our images.
img_width, img_height = 300, 300

# training data path
train_data_dir = 'data/train'


# validation set path
validation_data_dir = 'data/test'

# number of training samples
nb_train_samples = 36181

# number of validation set samples
nb_validation_samples = 100

# number of epochs
epochs = 15

# batch size
batch_size = 20

# preprocess training & validation sets - to be run once only
# preprocessing(img_width, img_height, train_data_dir + '/abnormal')
# preprocessing(img_width, img_height, train_data_dir + '/normal')
# preprocessing(img_width, img_height, validation_data_dir + '/abnormal')
# preprocessing(img_width, img_height, validation_data_dir + '/normal')

in_lay = Input([img_width, img_height, 3])
base_pretrained_model = InceptionV3(input_shape =  [img_width, img_height, 3], include_top = False, weights = 'imagenet')

# make the pretrained model trainable
base_pretrained_model.trainable = True

# extract features from pretrained model and feed to attention layers
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to enhance model accuracy
attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
attn_layer = Conv2D(32, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(1,
                    kernel_size = (1,1),
                    padding = 'valid',
                    activation = 'sigmoid')(attn_layer)

# distribute to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same',
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = True
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)

# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.25)(gap)
dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
out_layer = Dense(1, activation = 'sigmoid')(dr_steps)
retina_model = Model(inputs = [in_lay], outputs = [out_layer])

retina_model.compile(optimizer = Nadam(lr=0.00005), loss = 'binary_crossentropy',
                           metrics = ['accuracy'])

# print model summary
retina_model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    rotation_range=360,
    preprocessing_function=preprocess_input,
  )

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# To reduce learning if learning stops - waits for 10 epochs for no improvement
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=10, min_lr=0.000005, verbose=1)

# Stores best model - change name if you are changing parameters
checkpointer = ModelCheckpoint(filepath="Best_Model_Checkpoint.hdf5", verbose=1, save_best_only=True)

# weights to address class imbalance in training set
weight = {0: 0.366, 1: 0.634}

retina_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    class_weight=weight,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size, verbose=2, callbacks=[reduce_lr, checkpointer])

