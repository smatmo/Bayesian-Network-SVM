function [y,dy] = softHinge(x, r)
%SOFTHINGE computes the soft hinge ;-)
%
% according to the paper
% [y,dy] = softHinge(x, gamma, r)
% calculate soft hinge y and derivative of the soft hinge dy
%

y = zeros(size(x));
sqrt2 = sqrt(2);
m2 = r;
m1 = r * (1-sqrt2);

%%% constant region
idx1 = x < m1;
y(idx1) = 0;

%%% circle region
idx2 = (x >= m1) & (x < m1 + r/sqrt2);
y(idx2) = m2 - sqrt(r^2 - (x(idx2) - m1).^2);
%y(idx2) = r - sqrt(2*r*(1-sqrt2) * (x(idx2)-r) - x(idx2).^2);

%%% linear region
idx3 = (x >= m1 + r/sqrt2);
y(idx3) = x(idx3);


%%% gradient required?
if nargout == 2,
    dy = zeros(size(x));
    
    %%% linear region
    dy(idx1) = 0;
    
    %%% circle region
    dy(idx2) = (x(idx2) - m1) ./ sqrt(r^2 - (x(idx2)-m1).^2);
    
    %%% constant region
    dy(idx3) = 1;
end
