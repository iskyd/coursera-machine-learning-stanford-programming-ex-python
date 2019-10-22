import numpy
import matplotlib.pyplot as plt

def featureNormalize(X):
    mu = numpy.mean(X, axis=0)
    sigma = numpy.std(X, axis=0, ddof=1)

    t = numpy.ones((len(X), 1))

    X = (X - (t * mu)) / (t * sigma)

    return X, mu, sigma


def computeCostMulti(X, y, theta):
    m = len(y)

    return (1 / (2 * m)) * (X @ theta - y).T @ (X @ theta - y)

def gradientDescentMulti(X, y, theta, alpha, iterations):
    J_history = numpy.zeros((iterations, 1))

    m = len(y)

    for i in range(iterations):
        theta = theta - alpha * (1 / m) * (((X @ theta) - y).T @ X).T

        J_history[i:] = computeCostMulti(X, y, theta)

    return theta, J_history

def normalEquations(X, y):
    theta = numpy.linalg.inv((X.T @ X)) @ (X.T @ y)

    return theta
 
def main():
    data = numpy.loadtxt('./ex1data2.txt', dtype='float', delimiter =',')
    X = data[:,:2]
    y = data[:,[2]]
    m = len(y)

    X, mu, sigma = featureNormalize(X)

    X = numpy.column_stack((numpy.ones((m, 1)), X))
    
    theta = numpy.zeros((3, 1))

    cost = computeCostMulti(X, y, theta)
    
    alpha = 0.1
    iterations = 1500
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, iterations)

    plt.xlabel('iterations')
    plt.ylabel('theta')
    plt.plot(range(iterations), J_history)
    plt.show()

    house_sqr_feet = 1650
    house_bdr_nr = 3

    data_house = numpy.matrix([house_sqr_feet, house_bdr_nr])
    t = numpy.ones((len(data_house), 1))
    data_house_norm = (data_house - (t * mu)) / (t * sigma)

    X_predict = numpy.column_stack((numpy.ones((len(data_house_norm), 1)), data_house_norm))

    predicted_gradient_desc = X_predict @ theta
    print('Estimated price with gradient descent for house of {} square feet with {} bedroom: {}'.format(house_sqr_feet, house_bdr_nr, predicted_gradient_desc))

    theta = normalEquations(X, y)

    predicted_gradient_desc = X_predict @ theta
    print('Estimated price for house with normal equations of {} square feet with {} bedroom: {}'.format(house_sqr_feet, house_bdr_nr, predicted_gradient_desc))


if __name__ == '__main__':
    main()