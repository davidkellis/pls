% Implements "Modified kernel algorithm #2" as described in Dayal and MacGregor's "Improved PLS Algorithms" paper,
% published in Journal of Chemometrics, Volume 11, Issue 1, pages 73?85, January 1997.
function [beta,W,P,Q,R] = pls2(X, Y, A);
W=[];R=[];P=[];Q=[];
[N, K] = size(X);
[N, M] = size(Y);

XY = X' * Y;
XX = X' * X;
for i = 1:A,
  if M == 1,
    w = XY;
  else
    [C,D]=eig(XY'*XY);
    q=C(:,find(diag(D)==max(diag(D))));

    w=(XY*q);
  end
  w = w / sqrt(w' * w);
  r = w;
  for j = 1:i-1,
    r = r - (P(:, j)' * w) * R(:, j);
  end
  tt = (r' * XX * r);
  p = (r' * XX)' / tt;
  q = (r' * XY)' / tt;
  XY = XY - (p * q') * tt;

  W=[W w];
  P=[P p];
  Q=[Q q];
  R=[R r];
end
beta = R*Q';
