from models.cnn import mini_XCEPTION
from utils.inference import generator
from utils.datasets import getData

X_train, y_train = getData('../datasets/fer2013/train')
X_test, y_test = getData('../datasets/fer2013/validation')


model, cnn = mini_XCEPTION((50, 50, 1), 7)
model.summary()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(generator(X_train, y_train),steps_per_epoch=5,
          epochs=500,validation_data=generator(X_test, y_test),
          validation_steps=5)

model.save(r'E:\AI\face\trained_models\emotion_models\emtion_mini_XCEPTION1.h5')