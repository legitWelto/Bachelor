function p1_adaptive(red,theta,eps_stop) % Nb = []; d = 2;
error_bound = 1;
c4n = [-1,-1;0,-1;-1,0;0,0;1,0;-1,1;0,1;1,1];
n4e = [1,2,4;4,3,1;3,4,7;7,6,3;4,5,8;8,7,4];
Db = [1,2;2,4;4,5;5,8;8,7;7,6;6,3;3,1];
n = 1;
for j = 1:red
  marked = ones(size(n4e,1),1);
  [c4n,n4e,Db] = rgb_refine(c4n,n4e,Db,marked);
end
tic
while error_bound > eps_stop
  %%% solve
  fNodes = setdiff(1:size(c4n,1),unique(Db));
  u = zeros(size(c4n,1),1);
  [s,m] = fe_matrices(c4n,n4e);
  b = m*f(c4n);
  u(fNodes) = s(fNodes,fNodes)\b(fNodes);
  %show_p1(c4n,n4e,Db,[],u,n); waitforbuttonpress;
  n = n+1;
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
toc
size(c4n,1)
n-1

endfunction
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
function val = f(x); val = ones(size(x,1),1); endfunction

function [c4n,n4e,Db] = rgb_refine(c4n,n4e,Db,marked)
% edges n4e(:,[1,3]) have to be longest edges
[edges,el2edges,Db2edges] = edge_data_2d(n4e,Db);
nC = size(c4n,1); nEdges = size(edges,1); tmp = 1;
markedEdges = zeros(nEdges,1);
markedEdges(reshape(el2edges(marked==1,[1 2 3]),[],1)) = 1;
while tmp > 0
  tmp = nnz(markedEdges);
  el2markedEdges = markedEdges(el2edges);
  el2markedEdges(el2markedEdges(:,1)+el2markedEdges(:,3)>0,2)=1;
  markedEdges(el2edges(el2markedEdges==1))= 1;
  tmp = nnz(markedEdges)-tmp;
end
newNodes = zeros(nEdges,1);
newNodes(markedEdges==1) = (1:nnz(markedEdges))'+nC;
newInd = newNodes(el2edges);
red = newInd(:,1) > 0 & newInd(:,2) > 0 & newInd(:,3) > 0;
blue1 = newInd(:,1) > 0 & newInd(:,2) > 0 & newInd(:,3) == 0;
blue3 = newInd(:,1) == 0 & newInd(:,2) > 0 & newInd(:,3) > 0;
green = newInd(:,1) == 0 & newInd(:,2) > 0 & newInd(:,3) == 0;
remain= newInd(:,1) == 0 & newInd(:,2) == 0 & newInd(:,3) == 0;
n4e_red = [n4e(red,1),newInd(red,[3 2]),...
newInd(red,[2 1]),n4e(red,3),...
newInd(red,3),n4e(red,2),newInd(red,1),...
newInd(red,:)];
n4e_red = reshape(n4e_red',3,[])';
n4e_blue1 = [n4e(blue1,2),newInd(blue1,2),n4e(blue1,1) ...
            n4e(blue1,2),newInd(blue1,[1 2]),...
            newInd(blue1,[2 1]),n4e(blue1,3)];
n4e_blue1 = reshape(n4e_blue1',3,[])';
n4e_blue3 = [n4e(blue3,1),newInd(blue3,3),newInd(blue3,2),...
            newInd(blue3,2),newInd(blue3,3),n4e(blue3,2),...
            n4e(blue3,3),newInd(blue3,2),n4e(blue3,2)];
n4e_blue3 = reshape(n4e_blue3',3,[])';
n4e_green = [n4e(green,2),newInd(green,2),n4e(green,1),...
            n4e(green,3),newInd(green,2),n4e(green,2)];
n4e_green = reshape(n4e_green',3,[])';
n4e = [n4e(remain,:);n4e_red;n4e_blue1;n4e_blue3;n4e_green];
newCoord =.5*(c4n(edges(markedEdges==1,1),:)...
          +c4n(edges(markedEdges==1,2),:));
c4n = [c4n;newCoord];
newDb = newNodes(Db2edges); ref = newDb>0; Db_old = Db(~ref,:);
Db_new = [Db(ref,1),newDb(ref),newDb(ref),Db(ref,2)];
Db_new = reshape(Db_new',2,[])'; Db = [Db_old;Db_new];
endfunction

function [edges,el2edges,Db2edges] = edge_data_2d(n4e,Db)
nE = size(n4e,1); nEdges = 3*nE; nDb = size(Db,1);
edges = [reshape(n4e(:,[2 3,1 3,1 2])',2,[])';Db];
[edges,~,edgeNumbers] = unique(sort(edges,2),'rows','first');
el2edges = reshape(edgeNumbers(1:nEdges),3,[])';
Db2edges = reshape(edgeNumbers(nEdges+(1:nDb))',1,[])';
end

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

function show_p1(c4n,n4e,Db,Nb,u,n)
d = size(c4n,2);
if d == 1
  plot(c4n(n4e),u(n4e));
elseif d == 2
  subplot(1,2,1.5 +0.5*(-1)^n);  
  trisurf(n4e,c4n(:,1),c4n(:,2),u);
  view(45,45);
  #subplot(2,6,2*n);
  #trisurf(n4e,c4n(:,1),c4n(:,2),u);
  #view(0,90);
  #subplot(1,2,1.5 +0.5*(-1)^n);
  #colormap(white);
  #trisurf(n4e,c4n(:,1),c4n(:,2),zeros(size(c4n,1),1));
  #view(0,90);
elseif d == 3
  trisurf([Db;Nb],c4n(:,1),c4n(:,2),c4n(:,3),u);
end
endfunction