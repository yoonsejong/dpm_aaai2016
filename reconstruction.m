function [ X_hat, X_hat_var] = reconstruction( model )
% Reconstruction
N = size(model.mZ,2);
D = size(model.mMU,1);
X_hat = model.mW * model.mZ + repmat(model.mMU, [1, N]);
X_hat_var = zeros(D,N);

for i = 1 : D
    for j= 1 : N
        
        X_hat_var(i,j) = model.vMU(i,1) + model.mW(i,:) * model.vZ(:,:,j) ...
            * model.mW(i,:)' + model.mZ(:,j)' * diag(model.vW(i,:)) ...
            * model.mZ(:,j) + trace(model.vZ(:,:,j) * diag(model.vW(i,:)));
    end
end
   
end

