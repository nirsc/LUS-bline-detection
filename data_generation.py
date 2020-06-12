from sklearn.datasets import make_regression
import matplotlib.pyplot as plt




def generate_data(num_lines,n_samples,noise_levels, biases):
    lines = []
    for i in range(num_lines):
        n = n_samples[i]
        noise = noise_levels[i]
        bias = biases[i]
        x,y,coef = make_regression(n_samples = n, n_features= 1, noise = noise, bias = bias, coef = True)

        # make classification seems to create lines with mainly
        #positive slopes. This helps make things more symmetric
        if i % 2 :
            real_y = -coef*x -bias
            y = -y
        else:
            real_y = coef*x + bias
        lines += [(x,y,real_y)]

    return lines

noises = [1*i for i in range(3)]
biases = [0,10, 50]
n_samples = [30]*3
lines = generate_data(3,n_samples,noises,biases)
colors = ['r','y','g']
for i in range(len(n_samples)):
    x = lines[i][0]
    y = lines[i][1]
    real_y = lines[i][2]
    plt.scatter(x,y,color=colors[i])
    plt.plot(x,real_y)
plt.show()