# -*- coding: utf-8 -*-
"""Integrales metodos de aproximacion.ipynb

### Regla de los trapecios
"""

import sympy
import matplotlib.pyplot as plt
import numpy as np

x = sympy.symbols("x")
a = -1
b = 1
intervals = 3
h = (b-a)/intervals

y = 5*x**4-x**3+2*x**2+x
sumatory = 0
f_a = y.evalf(subs={x: a})
x_a = a
x_i = a + h
f_b = 0

for i in range(intervals):
    f_b = y.evalf(subs={x: x_i})
    sumatory += (f_a+f_b)*(x_i-x_a)/2 
    x_a = x_i
    x_i += h
    f_a = f_b
print("El area aproximada de la funcion y entre", a , "y", b, "con", intervals, "trapecios es de ", sumatory)

area_under_curve = sympy.integrate(y, (x,a,b))
print("El area bajo la curva verdadera es", area_under_curve)

print("El error de truncamiento es de", area_under_curve - sumatory)
x_value = np.linspace(a,b,intervals+1)
y_value = np.array([y.evalf(subs={x:i}) for i in x_value], dtype=float)
x_real = np.linspace(a,b,1000)
y_real = np.array([y.evalf(subs={x:i}) for i in x_real], dtype=float)

plt.fill_between(x_value, y_value, label="Aproximation")
plt.title(f"{y}")
plt.plot(x_real,y_real, "r", linewidth=4, label="Real function")
plt.legend()
plt.show()

import sympy
import matplotlib.pyplot as plt
import numpy as np

x = sympy.symbols("x")
a = 0
b = 1
intervals = 3
h = (b-a)/intervals

y = np.e**x
sumatory = 0
f_a = y.evalf(subs={x: a})
x_a = a
x_i = a + h
f_b = 0

for i in range(intervals):
    f_b = y.evalf(subs={x: x_i})
    sumatory += (f_a+f_b)*(x_i-x_a)/2 
    x_a = x_i
    x_i += h
    f_a = f_b
print("El area aproximada de la funcion y entre", a , "y", b, "con", intervals, "trapecios es de ", sumatory)

area_under_curve = round(sympy.integrate(y, (x,a,b)),10)
print("El area bajo la curva verdadera es", area_under_curve)

print("El error de truncamiento es de", area_under_curve - sumatory)
x_value = np.linspace(a,b,intervals+1)
y_value = np.array([y.evalf(subs={x:i}) for i in x_value], dtype=float)
x_real = np.linspace(a,b,1000)
y_real = np.array([y.evalf(subs={x:i}) for i in x_real], dtype=float)

plt.fill_between(x_value, y_value, label="Aproximation")
plt.title(f"{y}")
plt.plot(x_real,y_real, "r", linewidth=4, label="Real function")
plt.legend()
plt.show()

import sympy
import matplotlib.pyplot as plt
import numpy as np

x = sympy.symbols("x")
a = 0
b = 2
intervals = 4
h = (b-a)/intervals

y = sympy.sin(x)
sumatory = 0
f_a = y.evalf(subs={x: a})
x_a = a
x_i = a + h
f_b = 0

for i in range(intervals):
    f_b = y.evalf(subs={x: x_i})
    sumatory += (f_a+f_b)*(x_i-x_a)/2 
    x_a = x_i
    x_i += h
    f_a = f_b
print("El area aproximada de la funcion y entre", a , "y", b, "con", intervals, "trapecios es de ", sumatory)

area_under_curve = round(sympy.integrate(y, (x,a,b)),10)
print("El area bajo la curva verdadera es", area_under_curve)

print("El error de truncamiento es de", area_under_curve - sumatory)
x_value = np.linspace(a,b,intervals+1)
y_value = np.array([y.evalf(subs={x:i}) for i in x_value], dtype=float)
x_real = np.linspace(a,b,1000)
y_real = np.array([y.evalf(subs={x:i}) for i in x_real], dtype=float)

plt.fill_between(x_value, y_value, label="Aproximation")
plt.title(f"{y}")
plt.plot(x_real,y_real, "r", linewidth=4, label="Real function")
plt.legend()
plt.show()

import sympy
import matplotlib.pyplot as plt
import numpy as np

x = sympy.symbols("x")

a = 2
b = 3
intervals = 4
h = (b-a)/intervals

y = sympy.sin(x)
sumatory = 0
f_a = y.evalf(subs={x: a})
x_a = a
x_i = a + h
f_b = 0

for i in range(intervals):
    f_b = y.evalf(subs={x: x_i})
    sumatory += (f_a+f_b)*(x_i-x_a)/2 
    x_a = x_i
    x_i += h
    f_a = f_b
print("El area aproximada de la funcion y entre", a , "y", b, "con", intervals, "trapecios es de ", sumatory)

area_under_curve = round(sympy.integrate(y, (x,a,b)),10)
print("El area bajo la curva verdadera es", area_under_curve)

print("El error de truncamiento es de", area_under_curve - sumatory)
x_value = np.linspace(a,b,intervals+1)
y_value = np.array([y.evalf(subs={x:i}) for i in x_value], dtype=float)
x_real = np.linspace(a,b,1000)
y_real = np.array([y.evalf(subs={x:i}) for i in x_real], dtype=float)

plt.fill_between(x_value, y_value, label="Aproximation")
plt.title(f"{y}")
plt.plot(x_real,y_real, "r", linewidth=4, label="Real function")
plt.legend()
plt.show()

"""### Simpson 1/3"""

#Regla de Simpson 1/3
import sympy
x = sympy.symbols("x")
y =  sympy.sin(x) 
a = 0
b = 2
intervals = 4
h = (b - a) / intervals
x_medio = (b-a)/2
#Sumamos los extremos
sumatory = y.evalf(subs={x:a}) + y.evalf(subs={x:b})
    
for i in range(1,intervals):
  x_i = a + i*h
  """Si es par se multiplica por 2 y si es impar se multiplica por 4
        """
  if i%2 == 0:
    sumatory +=  2 * y.evalf(subs={x:x_i})
  else:
    sumatory += 4 * y.evalf(subs={x:x_i})
    
sumatory = round(sumatory * (h/3),20) #Resultado de la aproximacion

print("El area aproximada de la funcion y entre", a , "y", b, "con simpson 1/3 para", intervals, "intervalos es de ", sumatory)

area_under_curve = round(sympy.integrate(y, (x,a,b)),10)
print("El area bajo la curva verdadera es", area_under_curve)

print("El error de truncamiento es de", area_under_curve - sumatory)

x_value = np.linspace(a,x_medio,3)
x_value_1 = np.linspace(x_medio,b ,3)
x_real = np.linspace(a,b,1000)
y_real = np.array([y.evalf(subs={x:i}) for i in x_real], dtype=float)
modelo = np.poly1d (np.polyfit (x_real, y_real, 2))
print(modelo)
y_value = np.array([modelo.evalf(subs={x:i}) for i in x_value], dtype=float)
y_value_1 = np.array([modelo.evalf(subs={x:i}) for i in x_value_1], dtype=float)



#add línea polinomial ajustada al diagrama de dispersión
polilínea = np.linspace (a,x_medio, 3)
plt.scatter (x_value, y_value)
plt.plot (polilínea, modelo (polilínea))
plt.show ()

print(modelo)

plt.fill_between(x_value, y_value, label="Aproximation")
plt.fill_between(x_value_1, y_value_1, label="Aproximation")
plt.title(f"{y}")
plt.plot(x_real,y_real, "r", linewidth=4, label="Real function")
plt.legend()
plt.show()
