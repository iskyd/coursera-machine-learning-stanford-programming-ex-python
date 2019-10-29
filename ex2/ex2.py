import numpy
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def plotData(X, y):
    pos = X[numpy.where(y==1)[0]]
    neg = X[numpy.where(y==0)[0]]

    plt.plot(pos.T[0], pos.T[1], 'r+', label='Admitted')
    plt.plot(neg.T[0], neg.T[1], 'ro', label='Not Admitted')

    plt.ylabel('Exam 2')
    plt.xlabel('Exam 1')

    plt.show()

def plotDecisionBoundary(theta, X, y):
    pos = X[numpy.where(y==1)[0]]
    neg = X[numpy.where(y==0)[0]]

    plt.plot(pos.T[0], pos.T[1], 'r+', label='Admitted')
    plt.plot(neg.T[0], neg.T[1], 'ro', label='Not Admitted')

    plt.ylabel('Exam 2')
    plt.xlabel('Exam 1')

    if X.shape[1] <= 3:
        plot_x = numpy.array([numpy.min(X.T[0]) - 2, numpy.max(X.T[0]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        plt.plot(plot_x, plot_y)

    plt.show()

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def costFunction(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))

    h_theta = sigmoid(X @ theta)
    J = (1 / m) * ((-y.T @ numpy.log(h_theta)) - (1 - y).T @ numpy.log(1 - h_theta))

    grad = (1 / m) * (h_theta - y).T @ X
    grad = grad.reshape((n, 1))

    return J, grad

def main():
    data = numpy.loadtxt('./ex2data1.txt', dtype='float', delimiter =',')
    X = data[:,:2]
    y = data[:,[2]]

    m = y.size

    plotData(X, y)

    X = numpy.column_stack((numpy.ones((m, 1)), X))
    m, n = X.shape

    initial_theta = numpy.zeros((n, 1))

    J, _grad = costFunction(initial_theta, X, y)

    print(J)

    result = optimize.minimize(
        fun = costFunction, 
        x0 = initial_theta, 
        args= (X, y), 
        method = 'TNC', 
        jac = True,
        options={'maxiter' : 400}
    )

    print('Number of iterations: {} - Number of evaluation: {}'.format(result.nit, result.nfev))
    print(result.x)

    optimal_theta = result.x

    tmp = numpy.array([1, 45, 85])
    prob = sigmoid(tmp @ optimal_theta)

    print('Probability of student with exam 1 score {} and exam 2 score {} : {}'.format(45, 85, prob))

    if prob > 0.5:
        print('Student admitted')
    else :
        print('Student not admitted')

    plotDecisionBoundary(optimal_theta, X[:,1:3], y)

    p = sigmoid(X @ optimal_theta) >= 0.5
    p = p.reshape((m, 1))
    p = p.astype(int)

    print('Train Accuracy: {}', numpy.mean(p == y) * 100)
    print('Expected accuracy (approx): 89.0')

if __name__ == '__main__':
    main()