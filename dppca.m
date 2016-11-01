function model = dppca( X, M, V, E, varargin )
% DPPCA      Distributed Probablistic PCA (D-PPCA)
% 
% Description
%  Solve probabilistic PCA problem in a distributed way. The network has 
%  max(V) nodes. We assume the network is connected. This function only 
%  simulates parameter broadcasts. Local computation is done by dppca_local 
%  function. NaN elements in X are considered as missing values.
%
% Input
%  X     : D x N matrix for full data from all nodes (N=N1+N2+...+NJ)
%  M     : Scalar of projection dimension
%  V     : N x 1 vector for each observation's source (node affiliation)
%  E     : J x J adjacency matrix where J = max(V)
%  [Optional Parameters]
%  InitModel  : D-PPCA model to set initial parameter (Def: random)
%  Threshold  : Scalar convergence criterion (Def: 1e-5)
%  ShowObjPer : If > 0, print out objective every specified iteration.
%               If 0, nothing will be printed. (Def: 1)
%  MaxIter    : Maximum iterations (Def: 1000)
%  ZeroMean   : True if we enforce the mean to be zero. (Def: false)
%  Eta        : Scalar of learning rate (Def: 10)
%
% Output
%  model = structure(W, MU, VAR, ...);
%  W        : J cells; D x M projection matrices for J nodes
%  MU       : J cells; D x 1 vector sample means for J nodes
%  VAR      : J cells; Scalar estimated variances for J nodes
%  EZ       : J cells; M x N matrix, mean of N latent vectors
%  EZZt     : J cells; M x M x N cube, covariance of N latent vectors
%  eITER    : Iterations took
%  eTIME    : Elapsed time
%  objArray : Objective function value change over iterations
%  LAMBDAi  : J cells; D x M matrix Lagrange multipliers
%  GAMMAi   : J cells; D x 1 vector Lagrange multipliers
%  BETAi    : J cells; Scalar       Lagrange multipliers
%
% Implemented
%  by     Sejong Yoon (sjyoon@cs.rutgers.edu)
%  on     2011.10.07 (last modified on 2015/03/19)
%
% References
%  [1] M.E. Tipping and C.M. Bishop, Probablistic principal component 
%      analysis, J. Royal Statistical Society B 21(3), pp. 611-622, 1999.
%  [2] Probablistic Modeling Toolkit 3, pmtk3.googlecode.com
%  [3] P.A. Forero, A. Cano and G.B. Giannakis, Distributed clustering
%      using wireless sensor networks, IEEE J. Selected Topics in Signal 
%      Processing 5(4), August 2011.
%  [4] S. Yoon and V. Pavlovic, Distributed Probabilistic Learning for 
%      Camera Networks with Missing Data, NIPS 25, 2012.

% Check required arguments
assert(nargin >= 4, 'Please specify at least X, M, V and E.');

% D dimensions x N samples
[D, N] = size(X);

% J = number of nodes
J = max(V);

% Check graph is valid
[r,c] = size(E);
assert(r == c, 'Adjacency matrix is not square!');
assert(sum(sum(abs(E' - E))) == 0, 'Graph should be indirectional!');
if J > 1
    assert(r == J, 'Adjacency matrix size does not match number of nodes!');
end

%--------------------------------------------------------------------------
% Parse optional parameters
p = inputParser;
p.StructExpand = false;

W = cell(J,1);
MU = cell(J,1);
VAR = cell(J,1);
for j = 1 : J
    W{j} = orth(randn(D, M));
    MU{j} = zeros(D, 1);
    VAR{j} = 1;
end
defaultMODEL = structure(W, MU, VAR);
defaultTHRESH = 1e-5;
defaultITER = 1;
defaultMaxIter = 1000;
defaultZeroMean = false;
defaultETA = 10;

addParameter(p,'InitModel',defaultMODEL);
addParameter(p,'Threshold',defaultTHRESH,@isnumeric);
addParameter(p,'ShowObjPer',defaultITER,@isnumeric);
addParameter(p,'MaxIter',defaultMaxIter);
addParameter(p,'ZeroMean',defaultZeroMean);
addParameter(p,'Eta',defaultETA,@isnumeric);

parse(p,varargin{:});

% Initialize parameters
model_init = p.Results.InitModel;
THRESH     = p.Results.Threshold;
iter_obj   = p.Results.ShowObjPer;
COUNTER_MAX = p.Results.MaxIter;
ZeroMean   = p.Results.ZeroMean;
ETA        = p.Results.Eta;

assert(ETA > 0, 'Learning rate (ETA) should be positive!');

% Check validity of initilaization
if (isfield(model_init, 'W') && ~iscell(model_init.W)) || ...
    (isfield(model_init, 'MU') && ~iscell(model_init.MU)) || ...
    (isfield(model_init, 'VAR') && ~iscell(model_init.VAR))
    error('Invalid initialization: please specify distributed model');
end

%--------------------------------------------------------------------------
% We need to broadcast parameters & Lagrange multipliers to neighbors. 
% Here, we use global variables. In real settings, sensors should transmit 
% them over network.
global Wi MUi PRECi oWi oMUi oPRECi;
global LAMBDAi GAMMAi BETAi;

% Local variables
global EZ EZZt;
global Bj MISSj;

% To reduce unnecessary memory copy
global Xj;

%--------------------------------------------------------------------------
% Build Xi for speed up
Xj = cell(J,1);
for i = 1 : J
    Xj{i} = X(:, V == i);
end

% Find i-th node's neighbor set Bi in advance to speed up
Bj = cell(J,1);
for i = 1 : J
    if r == 0
        Bj{i} = [];
    else
        Bj{i} = find(E(i,:) > 0);
    end
end

% Local parameters and auxiliary variables defined here for simplicity.
% In the real environment, these variables reside in each local sensor.
% Initialize parameters
for i = 1 : J
    Wi{i} = model_init.W{i};
    MUi{i} = model_init.MU{i};
    PRECi{i} = 1./model_init.VAR{i};
end

% Initialize Lagrange multipliers. Each edge of each node has a multiplier.
for i = 1 : J
    LAMBDAi{i} = zeros(D, M);
    GAMMAi{i} = zeros(D, 1);
    BETAi{i} = 0;
end

% Initialize latent variables
EZ = cell(J, 1);
EZZt = cell(J, 1);

% Build MISSi for speed up
MISSj = cell(J,1);
for i = 1 : J
    MISSj{i} = isnan(Xj{i});
end

% Learning rate
ETAhalf = ETA * 0.5;

% Initialize objective function - Lagrangian (we are minimizing this)
oldObjLR = realmax;
objArray = zeros(COUNTER_MAX, J+1); % last one is reserved for total
Fi = zeros(J, 1);

%--------------------------------------------------------------------------
% Prepare performance measures
converged = 0;
counter = 1;
tic;

% Main loop
while counter <= COUNTER_MAX
    %----------------------------------------------------------------------
    % Temporarily store parameters to simulate broadcasting and
    % synchronization. All nodes should update before sending their values.
    oWi = Wi;
    oMUi = MUi;
    oPRECi = PRECi;
    
    %----------------------------------------------------------------------
    % In each node: Update parameters locally
    for i = 1 : J
        Fi(i) = dppca_local( M, i, ETA, ZeroMean );
    end
    
    %----------------------------------------------------------------------
    % In each node: Update Lagrange multipliers
    for i = 1 : J
        Bi = Bj{i};
        for j = 1:length(Bi)
            LAMBDAi{i} = LAMBDAi{i} + ( ETAhalf * (Wi{i} - Wi{Bi(j)}) );
            GAMMAi{i} = GAMMAi{i} + ( ETAhalf * (MUi{i} - MUi{Bi(j)}) );
            BETAi{i} = BETAi{i} + ( ETAhalf * (PRECi{i} - PRECi{Bi(j)}) );
        end
    end
    
    %----------------------------------------------------------------------
    % Stopping criterion checkpoint
        
    % Compute objective
    objLR = 0;
    for i = 1 : J
        objLRi = Fi(i);
        Bi = Bj{i};
        for j = 1:length(Bi) 
            objLRi = objLRi ...
                + trace(LAMBDAi{i}' * (Wi{i} - Wi{Bi(j)})) ...
                + (GAMMAi{i}' * (MUi{i} - MUi{Bi(j)})) ...
                + (BETAi{i} * (PRECi{i} - PRECi{Bi(j)})) ...
                + ETAhalf * norm(Wi{i} - Wi{Bi(j)},'fro')^2 ...
                + ETAhalf * norm(MUi{i} - MUi{Bi(j)},'fro')^2 ...
                + ETAhalf * (PRECi{i} - PRECi{Bi(j)})^2;
        end
        objArray(counter, i) = objLRi;
        objLR = objLR + objLRi;
    end
    objArray(counter,J+1) = objLR;
    relErr = (objLR - oldObjLR) / abs(oldObjLR);
    oldObjLR = objLR;
    
    % Show progress if requested
    if iter_obj > 0 && mod(counter, iter_obj) == 0
        fprintf('Iter = %d:  Cost = %f (rel %3.2f%%), RMS = %f, MSA = %.2e (J = %d, ETA = %f)\n', ...
            counter, objLR, relErr*100, ...
            calc_ppca_rms(Xj, Wi, EZ, MUi), ...
            calc_ppca_max_ssa(oWi, Wi), ...
            J, ETA);
    end
    
    % Check whether it has converged
    if abs(relErr) < THRESH
        converged = 1;
        break;
    end

    % Increase counter
    counter = counter + 1;
end

% Check convergence
if converged ~= 1
    fprintf('Could not converge within %d iterations.\n', COUNTER_MAX);
end

% Compute performance measures
eTIME = toc;
eITER = counter;

% Fill in remaining objective function value slots.
if counter < COUNTER_MAX
    % fill in the remaining objArray
    for i = 1:J+1
        objArray(counter+1:COUNTER_MAX,i) = ...
            repmat(objArray(counter,i), [COUNTER_MAX - counter, 1]);
    end
end

% Assign return values
W = cell(J,1);
MU = cell(J,1);
VAR = cell(J,1);
for i = 1 : J
    W{i} = Wi{i};
    MU{i} = MUi{i};
    VAR{i} = 1./PRECi{i};
end

% Create structure
model = structure( ...
    W, MU, VAR, ...
    EZ, EZZt, ...
    eITER, eTIME, objArray, ...
    LAMBDAi, GAMMAi, BETAi);

% Clean up 
clearvars -global Xj;
clearvars -global Wi MUi PRECi oWi oMUi oPRECi;
clearvars -global EZ EZZt;
clearvars -global Bj MISSj;
clearvars -global LAMBDAi GAMMAi BETAi;

clearvars -except model;

end

function [F_new] = dppca_local( M, i, ETA, isMeanZero )
% DPPCA_LOCAL  D-PPCA Local Update
% 
% Input
%  M     : Projected dimension
%  i     : Current node index
%  ETA   : Scalar Learning ratio
%  isMeanZero : True if we don't update MUi
%
% Output
%  F_new    : 1 x 1 scalar computed optimization forumla (first term only)

% Parameters and latent space variables: Not that only the three model 
% parameters need to be transmitted. Other variables are defined as global
% just for simple and easy-to-understand implementation.
global Wi MUi PRECi oWi oMUi oPRECi;
global LAMBDAi GAMMAi BETAi;
global EZ EZZt;
global Bj MISSj;
global Xj;

% Take i-th node
Xi = Xj{i};
Bi = Bj{i};
MISSi = MISSj{i};

% Get size of this samples and ball of this node
[D, Ni] = size(Xi);
cBj = length(Bi);

% Initialize latent variables (for loop implementation)
EZn = zeros(M, Ni);
EZnZnt = zeros(M, M, Ni);

%--------------------------------------------------------------------------
% E-step

for n = 1 : Ni
    % Get indicies of available features
    DcI = (MISSi(:,n) == 0);    
    
    % Compute Mi = Wi'Wi + VARi*I first
    Wc = Wi{i}(DcI,:);
    Mi = Wc' * Wc + 1/PRECi{i} * eye(M);

    % E[Zn] = Mi^(-1) * Wi' * (Xin - MUi)
    % Currently M x N
    EZn(:,n) = Mi \ Wc' * (Xi(DcI,n) - MUi{i}(DcI));

    % E[z_n z_n'] = VAR * Mi^(-1) + E[z_n]E[z_n]'
    % Currently M x M
    EZnZnt(:,:,n) = inv(Mi) * (1/PRECi{i}) + EZn(:,n) * EZn(:,n)';
end

%--------------------------------------------------------------------------
% M-step

% Update Wi
W_new = zeros(D, M);
% One dimension at a time
for d = 1 : D
    % Get non-missing point indexes
    NcI = (MISSi(d,:) == 0);

    W_new1 = sum( EZnZnt(:,:,NcI), 3 ) * oPRECi{i} + 2*ETA*cBj*eye(M);
    W_new2 = ( (Xi(d,NcI) - oMUi{i}(d)) * EZn(:,NcI)' ) * oPRECi{i};
    W_new3 = 2 * LAMBDAi{i}(d,:);
    W_new4 = zeros(1, M);
    for j = 1:cBj
        W_new4 = W_new4 + (oWi{i}(d,:) + oWi{Bi(j)}(d,:));
    end
    
    % Update
    if sum(sum(W_new1)) < eps
        W_new(d,:) = zeros(1,M);
    else
        W_new(d,:) = (W_new2 - W_new3 + ETA * W_new4) / W_new1;
    end
end

% Update MUi
MU_new = zeros(D,1);
if ~isMeanZero
    for d = 1 : D
        % Get non-missing point indexes
        NcI = (MISSi(d,:) == 0);
        
        MU_new1 = sum(NcI) * oPRECi{i} + 2*ETA*cBj;
        MU_new2 = oPRECi{i} * sum( Xi(d,NcI) - W_new(d,:) * EZn(:,NcI), 2 );
        MU_new3 = 2 * GAMMAi{i}(d);
        MU_new4 = 0;
        for j = 1 : cBj
            MU_new4 = MU_new4 + ( oMUi{i}(d) + oMUi{Bi(j)}(d) );
        end
        
        % Update
        MU_new(d) = (MU_new2 - MU_new3 + ETA * MU_new4) / MU_new1;
    end
end

% Update PRECi (by solve for VARi^(-1))
PREC_new1 = 2 * ETA * cBj;
PREC_new21 = 2 * BETAi{i};
PREC_new22 = 0;
for j = 1:cBj
    PREC_new22 = PREC_new22 + ETA * (oPRECi{i} + oPRECi{Bi(j)});
end
PREC_new23 = 0;
PREC_new24 = 0;
PREC_new4 = 0;
for n = 1:Ni
    % Get indices of available features
    DcI = (MISSi(:,n) == 0);
    PREC_new4 = PREC_new4 + sum(DcI);
    Wc = W_new(DcI,:);
    
    PREC_new23 = PREC_new23 + EZn(:,n)' * Wc' * (Xi(DcI,n) - MU_new(DcI));
    PREC_new24 = PREC_new24 + 0.5 * ( norm( Xi(DcI,n) - MU_new(DcI), 2 )^2 ...
        + trace( EZnZnt(:,:,n) * (Wc' * Wc) ) );
end
PREC_new2 = PREC_new21 - PREC_new22 - PREC_new23 + PREC_new24;
PREC_new3 = -PREC_new4 / 2;
PREC_new = roots([PREC_new1, PREC_new2, PREC_new3]);

% We follow larger, real solution.
if length(PREC_new) > 1
    PREC_new = max(PREC_new);
end
if abs(imag(PREC_new)) ~= 0i
    error('No real solution!');
    % We shouldn't reach here since both solutions are not real...
end
if PREC_new < 0
    error('Negative precicion!');
end

% Compute data log likelihood (we don't need to compute constant)
obj_val1 = 0;
obj_val2 = 0;
obj_val3 = 0;
obj_val4 = 0;
obj_val5 = 0;
for n = 1:Ni
    Id = (MISSi(:,n) == 0);
    Xc = Xi(Id,n);
    MUc = MU_new(Id);
    Wc = W_new(Id,:);        

    obj_val1 = obj_val1 + 0.5 * trace(EZnZnt(:,:,n));
    obj_val2 = obj_val2 + 0.5 * sum(Id) * log(2 * pi * PREC_new);
    obj_val3 = obj_val3 + (PREC_new/2) * norm(Xc - MUc, 2).^2;
    obj_val4 = obj_val4 + (PREC_new/2) * trace(EZnZnt(:,:,n) * (Wc' * Wc));
    obj_val5 = obj_val5 + PREC_new * EZn(:,n)' * Wc' * (Xc - MUc);
end
F_new = obj_val1 - obj_val2 + obj_val3 + obj_val4 - obj_val5;

% Update parameter values
Wi{i} = W_new;
MUi{i} = MU_new;
PRECi{i} = PREC_new;

EZ{i} = EZn;
EZZt{i} = EZnZnt;

end
