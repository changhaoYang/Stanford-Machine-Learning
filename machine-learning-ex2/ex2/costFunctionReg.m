function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
L = 0;
for i=2:length(theta)
	L = L + lambda/(2*m) * theta(i)^2;
end

h = sigmoid(X*theta); %m*1矩阵

temp = - log(h) .* y - log(1 - h) .* (1-y); %3*1矩阵
J = 1/m * sum(temp) + L;  %一个常数

temp2 = 1/m * (X' * (h - y));
grad = 1/m * (X' * (h - y)) + lambda/m * theta;
grad(1) = temp2(1); 




% =============================================================

end
