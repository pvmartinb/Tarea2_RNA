from ODE2solver import *

f0=lambda x,y:1.
f1=lambda x,y:0.
f2=lambda x,y:-y
f=[f0,f1,f2]
y0=1
dy0=-0.5

model=ODE2solver(f,y0,dy0)

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
y=-0.5*np.sin(x)+np.cos(x)
plt.plot(x,y)
plt.show()