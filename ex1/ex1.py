import numpy
import matplotlib.pyplot as plt


def plot(X, y, marker_type):
    plt.plot(X, y, marker_type)
    plt.ylabel('profit')
    plt.xlabel('population size')
    plt.show()

def computeCost(X, y, theta):
    m = len(y)
    J = 0

    h = X @ theta

    # Could be written like this

    # for i in range(0, m):
    #     J = J + ((h[i] - y[i]) ** 2)
    
    # J = J / (2 * m)

    # or simply like

    J = (1 / (2 * m)) * numpy.sum(numpy.square(h - y))

    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = numpy.zeros((iterations, 1))
    theta_history = numpy.zeros((iterations, 2))

    for i in range(iterations):
        h = X @ theta
        theta = theta - (1 / m) * alpha * (X.T.dot((h - y)))
        theta_history[i:] = theta.T
        J_history[i:] = computeCost(X, y, theta)

    return theta, J_history, theta_history

def main():
    data = numpy.loadtxt('./ex1data1.txt', dtype='float', delimiter =',')
    X = data[:,[0]]
    y = data[:,[1]]
    m = len(y)

    plot(X, y, marker_type='rx')

    X = numpy.column_stack((numpy.ones((m, 1)), data[:,[0]]))

    theta = numpy.zeros((2, 1))

    J = computeCost(X, y, theta)
    print(J)
    J = computeCost(X, y, numpy.array([[-1.], [2.]]))
    print(J)

    alpha = 0.01
    iterations = 1500

    theta, J_history, theta_history = gradientDescent(X, y, theta, alpha, iterations)

    print(theta)

    
    plt.xlabel('iterations')
    plt.ylabel('theta')
    plt.plot(range(iterations), J_history)
    plt.show()

    h = X @ theta

    plt.plot(X[:,1], y, 'rx')
    plt.plot(X[:,1], h)
    plt.ylabel('profit')
    plt.xlabel('population size')
    plt.show()


if __name__ == '__main__':
    main()