import numpy as np
A = np.array([[52, 20, 25],
              [30, 50, 20],
              [10, 30, 55]])

b = np.array([4800, 5810, 5990])
x = np.zeros(3)

# Definir la tolerancia y el número máximo de iteraciones
tol = 1e-3
max_iterations = 100

# Función para aplicar el método de Jacobi
def jacobi(A, b, x, tol, max_iterations):
    n = len(b)
    x_new = np.zeros_like(x)
    
    for iteration in range(max_iterations):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Verificar si los resultados han convergido
        if np.allclose(x, x_new, atol=tol):
            print(f"Convergió después de {iteration + 1} iteraciones")
            return x_new
        
        # Actualizamos los valores de las variables para la siguiente iteración
        x = x_new.copy()
    
    print("Número máximo de iteraciones alcanzado")
    return x_new

# Ejecutar el método de Jacobi
solution = jacobi(A, b, x, tol, max_iterations)

print("Solución:")
print(solution)
