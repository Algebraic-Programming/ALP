import numpy2alp
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])

result = numpy2alp.add_numpy_arrays(a, b)
print("Result from C++:", result)  # Output: [11. 22. 33.]

numpy2alp.print_numpy_array(result)    # Output from C++: Vector contents: 11 22 33
