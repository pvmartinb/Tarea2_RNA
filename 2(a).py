from ODEsolver import *

f0=lambda x,y:x
f1=lambda x,y:x**2-y
f=[f0,f1]
y0=0

model=ODEsolver(f,y0)

model.add(Dense(10, activation='tanh',input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])

x=tf.linspace(-5,5,100)
history=model.fit(x,epochs=500,verbose=1)

import matplotlib.pyplot as plt
a=model.predict(x)
plt.plot(x,a)
plt.show()