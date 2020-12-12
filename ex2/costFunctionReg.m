function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta,1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
	z = theta'*X(i,:)';
	J = J + ( - ( y(i)*log(sigmoid(z) ) ) - ( ( 1 - y(i) )*log( 1 - sigmoid(z)) ));

end;

tmp = theta.^2;
sot = sum(tmp([2:end],1));

J = (J/m) + ((lambda/(2*m))*sot);

z = X * theta;
an = sigmoid(z) - y;

X_copy = zeros(size(X));
for i = 1:m
	X_copy(i,:) = X(i,:).*an(i,1);

end;

for i = 2:n
	grad(i,1) = ((sum(X_copy(:,i)))/m)+((lambda/m)*theta(i,1));

end;

grad(1,1) = ((sum(X_copy(:,1)))/m);


% =============================================================

end
