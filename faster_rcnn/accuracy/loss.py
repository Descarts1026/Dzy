import glob
import matplotlib.pyplot as plt

x_axis = []
y_axis =[]

with open("../results20210801-093529.txt","r",encoding="utf-8") as f:
    lline = f.readlines()[:100]
    for index,ll in enumerate(lline):
        x_axis.append(index)
        y_axis.append(float(ll.split("  ")[-2]))

plt.title('train loss')
plt.plot(x_axis , y_axis , color='green', label='training loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
