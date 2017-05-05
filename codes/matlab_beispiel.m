function val = minors(A)
if size(A,2) == 4
    val = A(:,1).*A(:,4)-A(:,2).*A(:,3);
elseif size(A,2) == 9
    val = zeros(size(A,1),10);
    val(:,1:9) = [A(:,5).*A(:,9)-A(:,6).*A(:,8),...
        A(:,4).*A(:,9)-A(:,6).*A(:,7),...
        A(:,4).*A(:,8)-A(:,5).*A(:,7),...
        A(:,2).*A(:,9)-A(:,3).*A(:,8),...
        A(:,1).*A(:,9)-A(:,3).*A(:,7),...
        A(:,1).*A(:,8)-A(:,2).*A(:,7),...
        A(:,2).*A(:,6)-A(:,3).*A(:,5),...
        A(:,1).*A(:,6)-A(:,3).*A(:,4),...
        A(:,1).*A(:,5)-A(:,2).*A(:,4)];
    val(:,10) = A(:,1).*val(:,1)-A(:,2).*val(:,2)+A(:,3).*val(:,3);
end