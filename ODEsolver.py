import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

import numpy as np

class ODEsolver(Sequential):
	def __init__(self, f, y0, **kwargs):
		super().__init__(**kwargs)
		self.f=f
		self.y0=y0
		self.loss_tracker = keras.metrics.Mean(name="loss")

	@property
	def metrics(self):
		return [self.loss_tracker]

	def train_step(self,data):
		batch_size=tf.shape(data)[0]
		x=tf.random.uniform((batch_size,1),minval=-5,maxval=5)

		with tf.GradientTape() as tape:
			with tf.GradientTape() as tape2:
				tape2.watch(x)
				y_pred=self(x, training=True)
			dy=tape2.gradient(y_pred,x)
			x_0=tf.zeros((batch_size,1))
			y_0=self(x_0,training=True)
			eq=self.f[0](x,y_pred)*dy-self.f[1](x,y_pred)
			ic=y_0-self.y0
			loss=keras.losses.mean_squared_error(0.,eq)+keras.losses.mean_squared_error(0.,ic)
			
		grads=tape.gradient(loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

		self.loss_tracker.update_state(loss)

		return {"loss": self.loss_tracker.result()}