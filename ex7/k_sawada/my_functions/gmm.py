import numpy as np


class GMM:
    def __init__(self, dimension, number, convergence=0.0001,
                 init_mu=None, init_sigma=None, init_pi=None):
        """class for em algorithm

        Args:
            dimension (int): dimension of input data
            number (int): number of gaussian distribution functions
            convergence (float, optional):
                convergence threshold. Defaults to 0.0001.
            init_mu (ndarray, axis=(numer, dimension)), optional):
                initaial average. Defaults to None.
            init_sigma (ndarray, axis=(numer, dimension, dimension)), optional):
                initial covariance matrix. Defaults to None.
            init_pi (ndarray, axis=(number), optional):
                initial coefficient. Defaults to None.
        """
        # initialize parameters
        if (init_mu is None):
            init_mu = np.random.random_sample((number, dimension))
        if (init_sigma is None):
            init_sigma = np.random.random_sample((number, dimension, dimension))
        if (init_pi is None):
            init_pi = np.random.random_sample(number)
            init_pi /= np.sum(init_pi)
        
        self.mu = init_mu
        self.sigma = init_sigma
        self.pi = init_pi
        self.dimension = dimension
        self.number = number
        self.convergence = convergence
        self.log_likelihoods = []
        
        # make gaussian distribution functions
        self._make_gaussian()
    
    def _make_gaussian(self):
        """construct probability density function list
        """
        self.gaussian = []
        for i in range(self.number):
            self.gaussian.append(
                GaussianDistribution(self.mu[i], self.sigma[i], self.pi[i]))
        
    def _log_likelihood(self, x):
        """calcurate log likelihood

        Args:
            x (ndarray, axis=(data, dimension)): input data

        Returns:
            float: log likelihood
        """
        p = np.zeros((self.number, len(x)))
        for i in range(self.number):
            p[i] = self.gaussian[i].calculate(x)
        return np.sum(np.log(np.sum(p, axis=0)))
        
    def em(self, data):
        """fit with em algorithm

        Args:
            data (ndarray, axis=(data, dimension)): input data
        """
        self.log_likelihoods = [self._log_likelihood(data)]
        
        while True:
            # E step
            gamma = np.zeros((len(data), self.number))
            for i in range(self.number):
                gamma[:, i] = self.gaussian[i].calculate(data)
            gamma = gamma / np.sum(gamma, axis=1)[:, np.newaxis]
            
            # M step
            n = np.sum(gamma, axis=0)
            mu_new = gamma.T @ data / n.reshape(self.number, 1)
            pi_new = n / np.sum(n)
            sigma_new = np.zeros_like(self.sigma)
            # NOTE: for 消すためにはどうすればいい?
            # 3次元配列の @ を考えるといける? a @ b において, aの最後の次元と bの最後から2番目の次元が合えば計算可能
            for k in range(self.number):
                sigma_new[k] = (data - mu_new[k]).T @ \
                    ((data - mu_new[k]) * gamma[:, k, np.newaxis]) / n[k]

            # update parameters
            self.mu = mu_new
            self.sigma = sigma_new
            self.pi = pi_new
            
            self._make_gaussian()
            self.log_likelihoods.append(self._log_likelihood(data))
            if (self.log_likelihoods[-1] - self.log_likelihoods[-2] <
                    self.convergence):
                break
        
        
class GaussianDistribution:
    def __init__(self, mu, sigma, pi=1):
        """Gaussian Distribution function

        Args:
            mu (ndarray or float, axis=(dimension)): average
            sigma (ndarray or float, axis=(dimension, dimension)):
                Covariance matrix
            pi (float):
                coeffieient of probability density function. Defaults to 1
        """
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        if sigma.ndim == 0:
            # if scalar given for sigma, covert to 2-dimensinal matrix
            self.dimension = 1
            self.sigma = np.reshape(self.sigma, (1, 1))
        else:
            self.dimension = len(sigma)
    
    def calculate(self, x):
        """returns probability density function value

        Args:
            x (ndarray(axis=(data, dimention)) or float): input data

        Returns:
            ndarray(axis=(data)) or float: probability
        """
        values = np.zeros(len(x))
        for j in range(len(x)):
            values[j] = self.pi * \
                np.exp(-1 / 2 * (x[j] - self.mu).reshape((1, self.dimension)) @
                       np.linalg.inv(self.sigma) @
                       (x[j] - self.mu).reshape((self.dimension, 1))) / \
                (np.power(2 * np.pi, self.dimension / 2) *
                 np.power(np.linalg.det(self.sigma), 1 / 2))
        return values
