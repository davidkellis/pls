% A comparison of nine PLS1 algorithms ( DOI: 10.1002/cem.1248 )
function [b,W,P,T,Q,R] = pls1(X, Y, A);
W=[];R=[];P=[];Q=[];T=[];U=[];
XY = X' * Y;
for a = 1:A
  w = XY;
  w = w/sqrt(w' * w);
  r = w;
  for j=1:a-1,
    r = r - (P(:, j)' * w) * R(:, j);
  end
  t = X * r;
  tt = t' * t;
  p = X' * t / tt;
  q = (r' * XY)' / tt; 
  XY = XY - (p * q') * tt; 
  W=[W w]; 
  P=[P p]; 
  T=[T t]; 
  Q=[Q q]; 
  R=[R r];
end
b = cumsum(R * diag(Q'), 2);