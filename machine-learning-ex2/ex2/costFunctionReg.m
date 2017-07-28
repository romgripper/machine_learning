function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
reg = theta(2:end, :);
reg = reg' * reg * lambda / 2 / m;
J = - (y' * log(h) + (1 .- y)' * log(1 .- h)) / m + reg;
d = h - y;
theta1 = theta;
theta1(1,1) = 0;
grad = X' * d / m + lambda * theta1 / m;





% =============================================================

end
