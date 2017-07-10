function p1_adaptive(c4n,n4e,Db) % Nb = []; d = 2;
eps_stop = 1e-2; error_bound = 1; theta = 0.5;
while error_bound > eps_stop
  %%% solve
  fNodes = setdiff(1:size(c4n,1),unique(Db));
  u = zeros(size(c4n,1),1);
  [s,m] = fe_matrices(c4n,n4e);
  b = m*f(c4n);
  u(fNodes) = s(fNodes,fNodes)\b(fNodes);
  trisurf(n4e,c4n(:,1),c4n(:,2),u); view(0,90); pause(.05)
  %%% estimate
  eta = comp_estimators(c4n,n4e,Db,u);
  error_bound = sqrt(sum(eta.^2))
  %%% mark
  marked = (eta>theta*max(eta));
  %%% refine
  if error_bound > eps_stop
    [c4n,n4e,Db] = rgb_refine(c4n,n4e,Db,marked);
  end
end
endfunction

function [s,m,m_lumped,vol_T] = fe_matrices(c4n,n4e)
[nC,d] = size(c4n); nE = size(n4e,1);
m_loc = (ones(d+1,d+1)+eye(d+1))/((d+1)*(d+2));
ctr = 0; ctr_max = (d+1)^2*nE;
I = zeros(ctr_max,1); J = zeros(ctr_max,1);
X_s = zeros(ctr_max,1); X_m = zeros(ctr_max,1);
m_lumped_diag = zeros(nC,1); vol_T = zeros(nE,1);
for j = 1:nE
  X_T = [ones(1,d+1);c4n(n4e(j,:),:)'];
  grads_T = X_T\[zeros(1,d);eye(d)];
  vol_T(j) = det(X_T)/factorial(d);
  for m = 1:d+1
    for n = 1:d+1
      ctr = ctr+1; I(ctr) = n4e(j,m); J(ctr) = n4e(j,n);
      X_s(ctr) = vol_T(j)*grads_T(m,:)*grads_T(n,:)';
      X_m(ctr) = vol_T(j)*m_loc(m,n);
    end
    m_lumped_diag(n4e(j,m)) = m_lumped_diag(n4e(j,m))...
                              +vol_T(j)/(d+1);
  end
end
s = sparse(I,J,X_s,nC,nC); m = sparse(I,J,X_m,nC,nC);
m_lumped = diag(m_lumped_diag);
endfunction