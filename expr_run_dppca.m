function [cm3, cm4] = expr_run_dppca(X, M, V, E, ETA, THRESHd, objfreq_d, m_init_d, ZeroMean, run_bayesian)

%% ------------------------------------------------------------------------
disp('*** Distributed Setting ***');

disp('* D-PPCA *');
cm3 = dppca(X, M, V, E, 'Eta', ETA, ...
    'InitModel', m_init_d, 'Threshold', THRESHd, 'ShowObjPer', objfreq_d, ...
    'ZeroMean', ZeroMean);

if run_bayesian
    disp('* D-BPCA *');
    cm4 = dbpca(X, M, V, E, 'Eta', ETA, ...
        'InitModel', m_init_d, 'Threshold', THRESHd, 'ShowObjPer', objfreq_d, ...
        'ZeroMean', ZeroMean);
else
    cm4 = cm3;
end

end
