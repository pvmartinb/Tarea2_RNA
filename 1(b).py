from AjusteFuncion import *

f=lambda x:1+2*x+4*x**3

model=AjusteFuncion(f)

model.add(Dense(10, activation='tanh',input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])

x=tf.linspace(-1,1,100)
history=model.fit(x,epochs=500,verbose=1)

import matplotlib.pyplot as plt
a=model.predict(x)
plt.plot(x,a)
y=f(x)
plt.plot(x,y)
plt.show()