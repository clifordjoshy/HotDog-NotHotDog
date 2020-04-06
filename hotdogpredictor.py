from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

model = load_model('trained_hotdog.h5')

IMG_SIZE = 224

direc = '' # Add prediction directory here

# for i in range(1,12):
#     image = load_img(direc+str(i)+'.jpg') 
    
#     image = img_to_array(image)
#     image = image/255
#     plt.imshow(image)
#     image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
   
#     pred = model.predict(np.array([image]))
#     plt.title(("not " if pred[0,0]>0 else "") + "hot dog")
#     plt.show()

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(direc,
            class_mode='binary',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=20,
            shuffle = False
            )

batch = generator.next()
for pic in batch[0]:
    plt.imshow(pic)
    pred = model.predict(np.array([pic]))
    plt.title(("not " if pred[0,0]>0 else "") + "hot dog")
    plt.show()

    
    