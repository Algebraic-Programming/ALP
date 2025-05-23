import numpy2alp
import numpy as np

# 1D example
a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
result1d = numpy2alp.add_numpy_arrays(a, b)
print("1D Result:", result1d)
numpy2alp.print_numpy_array(result1d)

# 2D example
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[10.0, 20.0], [30.0, 40.0]])
result2d = numpy2alp.add_numpy_arrays(A, B)
print("2D Result:\n", result2d)
numpy2alp.print_numpy_array(result2d)


# 3D example
AA = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]] )
BB = np.array([[[10.0, 20.0], [30.0, 40.0]],[[50.0, 60.0], [70.0, 80.0]]])
result3d = numpy2alp.add_numpy_arrays(AA, BB)
print("3D Result:\n", result3d)
numpy2alp.print_numpy_array(result3d)
