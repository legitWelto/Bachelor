function eta = comp_estimators(c4n,n4e,Db,u)
  [s4e,~,n4s,s4Db] = sides(n4e,Db,[]);
  [~,d] = size(c4n); nE = size(n4e,1); nS = size(n4s,1);
  eta_S = zeros(nS,1); eta_T_sq = zeros(nE,1);
  for j = 1:nE
    X_T = [ones(1,d+1);c4n(n4e(j,:),:)'];
    grads_T = X_T\[zeros(1,d);eye(d)];
    vol_T = det(X_T)/factorial(d);
    h_T = vol_T^(1/d);
    mp_T = sum(c4n(n4e(j,:),:),1)/(d+1);
    eta_T_sq(j) = h_T^2*vol_T*(f(mp_T));
    nabla_u_T = grads_T'*u(n4e(j,:));
    normal_times_area = -grads_T*vol_T*d;
    eta_S(s4e(j,:)) = h_T^((2-d)/2)*eta_S(s4e(j,:))...
    +normal_times_area *nabla_u_T;
  end
eta_S(s4Db) = 0;
eta_S_T_sq = sum(eta_S(s4e).^2,2);
eta = (eta_T_sq+eta_S_T_sq).^(1/2);
end
function val = f(x); val = ones(size(x,1),1);
endfunction

function [s4e,sign_s4e,n4s,s4Db,s4Nb,e4s] = sides(n4e,Db,Nb)
nE = size(n4e,1); d = size(n4e,2)-1;
nDb = size(Db,1); nNb = size(Nb,1);
if d == 2
  Tsides = [n4e(:,[2,3]);n4e(:,[3,1]);n4e(:,[1,2])];
else
  Tsides = [n4e(:,[2,4,3]);n4e(:,[1,3,4]);...
  n4e(:,[1,4,2]);n4e(:,[1,2,3])];
end
[n4s,i2,j] = unique(sort(Tsides,2),'rows');
s4e = reshape(j,nE,d+1); nS = size(n4s,1);
sign_s4e = ones((d+1)*nE,1); sign_s4e(i2) = -1;
sign_s4e = reshape(sign_s4e,nE,d+1);
[~,~,j2] = unique(sort([n4s;Db;Nb],2),'rows');
s4Db = j2(nS+(1:nDb)); s4Nb = j2(nS+nDb+(1:nNb));
e4s = zeros(size(n4s,1),2);
e4s(:,1) = mod(i2-1,nE)+1;
i_inner = setdiff(1:(d+1)*nE,i2);
e4s(j(i_inner),2) = mod(i_inner-1,nE)+1;
endfunction