function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hyp = X*theta;
res = hyp-y;
res1 = res.^2;

thet = theta(2:end);
thet = thet.^2;
a = sum(res1);
b = sum(thet);

J = ((1/(2*m))*a)+((lambda/(2*m))*b);

grad(1)=((sum(res))/m);
n = length(theta);

for i = 2:n
	c = ((lambda/m)*theta(i));
	res2 = res.*X(:,i);
	a = sum(res2);
	grad(i) = (a/m)+c;
endfor;


% =========================================================================

grad = grad(:);

end
