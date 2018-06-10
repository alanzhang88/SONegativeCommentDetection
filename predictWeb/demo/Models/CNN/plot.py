import sys
import numpy as np
import matplotlib.pyplot as plt

fieldName = sys.argv[1]
fieldValue = eval(sys.argv[2])
accuracy = eval(sys.argv[3])

x = np.array(list(range(len(accuracy))))
y = np.array(accuracy)
my_xticks = [str(e) for e in fieldValue]
plt.xticks(x,my_xticks)
plt.xlabel(fieldName)
plt.ylabel('accuracy')
plt.plot(x,y,'ro--')
plt.show()
