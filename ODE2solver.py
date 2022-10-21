import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

import numpy as np

class ODE2solver(Sequential):
	def __init__(self, f, y0, dy0, **kwargs):
		super().__init__(**kwargs)
		self.f=f
		self.y0=y0
		self.dy0=dy0
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
				with tf.GradientTape() as tape3:
					tape3.watch(x)
					y_pred=self(x, training=True)
				dy=tape3.gradient(y_pred,x)
			dy_0=dy[batch_size//2]
			dy2=tape2.gradient(dy,x)
			x_0=tf.zeros((batch_size,1))
			y_0=self(x_0,training=True)
			eq=self.f[0](x,y_pred)*dy2+self.f[1](x,y_pred)*dy-self.f[2](x,y_pred)
			ic=y_0-self.y0
			dic=dy_0-self.dy0
			loss=keras.losses.mean_squared_error(0.,eq)+keras.losses.mean_squared_error(0.,ic)+keras.losses.mean_squared_error(0.,dic)
			
		grads=tape.gradient(loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

		self.loss_tracker.update_state(loss)

		return {"loss": self.loss_tracker.result()}