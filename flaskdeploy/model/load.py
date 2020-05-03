from keras.models import model_from_json,load_model
import tensorflow as tf
def init():
  # json_file = open('model/model1.json','r')
  # loaded_model_json = json_file.read()
  # json_file.close()
  # loaded_model = model_from_json(loaded_model_json)
  loaded_model=load_model("model/CNN.best.from_scratch.hdf5")
  #load weights into new model
  loaded_model.load_weights("model/weights.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.get_default_graph()
  return loaded_model,graph