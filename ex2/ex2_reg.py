import numpy
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def plotData(X, y):
    pos = X[numpy.where(y==1)[0]]
    neg = X[numpy.where(y==0)[0]]

    plt.plot(pos.T[0], pos.T[1], 'r+', label='Admitted')
    plt.plot(neg.T[0], neg.T[1], 'ro', label='Not Admitted')

    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')

    plt.show()

def plotDecisionBoundary(theta, X, y):
    pos = X[numpy.where(y==1)[0]]
    neg = X[numpy.where(y==0)[0]]

    plt.plot(pos.T[0], pos.T[1], 'r+', label='Admitted')
    plt.plot(neg.T[0], neg.T[1], 'ro', label='Not Admitted')

    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')

    if X.shape[1] <= 3:
        plot_x = numpy.array([numpy.min(X.T[0]) - 2, numpy.max(X.T[0]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        plt.plot(plot_x, plot_y)
    else:
        # Here is the grid range
        u = numpy.linspace(-1, 1.5, 50)
        v = numpy.linspace(-1, 1.5, 50)

        z = numpy.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = numpy.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        plt.contourf(u, v, z, levels=[numpy.min(z), 0, numpy.max(z)], cmap='Greens', alpha=0.4)

    plt.show()

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def mapFeature(X1, X2):
    degree = 6
    out = numpy.ones(X1.size)

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = numpy.column_stack((out, (X1 ** (i-j)) * (X2 ** j)))
    
    return out

def costFunctionReg(theta, X, y, lmbd = 1):
    m = y.size
    h_theta = sigmoid(X @ theta)

    J = (1 / m) * (-y.T @ numpy.log(h_theta) - (1 - y).T @ numpy.log(1 - h_theta)) + (lmbd / (2 * m)) * (theta[1:theta.size]).T @ theta[1:theta.size]

    return J

def gradientDescent(theta, X, y, lmbd = 1):
    m, n = X.shape
    theta = theta.reshape((n, 1))

    h_theta = sigmoid(X @ theta)

    thetaZero = theta
    thetaZero[1] = 0

    grad = ((1 / m) * (h_theta - y).T @ X) + lmbd / m * thetaZero.T

    return grad.reshape((n, 1))

def main():
    data = numpy.loadtxt('./ex2data2.txt', dtype='float', delimiter =',')
    X = data[:,:2]
    y = data[:,[2]]

    plotData(X, y)

    X = mapFeature(X.T[0], X.T[1])
    
    m, n = X.shape

    initial_theta = numpy.zeros((n, 1))
    lmbd = 1 # lambda

    J = costFunctionReg(initial_theta, X, y, lmbd)
    grad = gradientDescent(initial_theta, X, y, lmbd)
    print(J)
    print(grad)

    test_theta = numpy.ones((n, 1))
    J = costFunctionReg(test_theta, X, y, 10)
    grad = gradientDescent(test_theta, X, y, 10)
    print(J)
    print(grad)

    result = optimize.minimize(
        fun = costFunctionReg, 
        x0 = initial_theta, 
        args= (X, y), 
        method = 'TNC', 
        jac = gradientDescent,
        options={'maxiter' : 400}
    )

    print('Number of iterations: {} - Number of evaluation: {}'.format(result.nit, result.nfev))
    print(result.x)

    optimal_theta = result.x

    plotDecisionBoundary(optimal_theta, X[:,1:], y)

    p = sigmoid(X @ optimal_theta) >= 0.5
    p = p.reshape((m, 1))
    p = p.astype(int)

    print('Train Accuracy: {}'.format(numpy.mean(p == y) * 100))
    print('Expected accuracy (approx): 83.1')


if __name__ == '__main__':
    main()