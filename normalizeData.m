function [X, Y] = normalizeData(Xs, Ys, a, b)
X = [];
Y = [];
if(nargin == 2),
   % normalization to 0
   miu = mean(Xs);
   sigma = std(Xs);
   X = (Xs - repmat(miu,[size(Xs,1) 1]))./ repmat(sigma,[size(Xs,1) 1]);
else
   if(nargin == 4)
       % normalization to [a, b]
       m = min(Xs);
       M = max(Xs);
       X = a+(b-a)*(Xs-repmat(m,[size(Xs,1) 1]))./repmat(M-m,[size(Xs,1) 1]);
   end
end
[X, ia, ib] = unique(X,'rows','stable');
Y=Ys(ia,:);
end