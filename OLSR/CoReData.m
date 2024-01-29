function [A,B,X] = CoReData(n,m,p,s)
%This code is used to generate data.

U0 = rand(n,n);
U=orth(U0);

V0 = rand(m,n);
V=orth(V0);
for i = 1:n
    xi(i) = 0.9^(i/2);
end
Sigma = diag(xi);

A = U*Sigma*(V');

I0      = randperm(n);
I       = I0(1:s);
X       = zeros(n,p);
X(I,:)	= randn(s,p);

B       = ((A')*X)';

end