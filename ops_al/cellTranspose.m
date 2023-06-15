%% % Transform the cell data from nxd to dxn

% Code author: Shiping Wang
% Email: shipingwangphd@gmail.com
% Date: May 23, 2019. 

function X1 = cellTranspose(X)

for ii = 1:length(X)
    tempMat = X{1,ii};
    tempMat = tempMat';
    X1{1,ii} = tempMat;
end

end