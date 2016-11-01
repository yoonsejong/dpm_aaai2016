function [  Ini_mZ, Ini_vZ, Ini_mW, Ini_vW, Ini_mMu, Ini_vMu, Ini_PW, Ini_PMu, Ini_PX] = Initialize(N,D,M)
% Initialize
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
reset(s,0);
Ini_mZ = randn(M,N);
Ini_mW = orth(rand(D, M));
Ini_mMu = zeros(D, 1);
Ini_vZ = repmat(eye(M, M), [1, 1, N]);
Ini_vW = ones(D, M);
Ini_vMu = ones(D, 1);
Ini_PX = 1;
Ini_PW = 1000 * ones(1,M);
Ini_PMu = 1000;
end

