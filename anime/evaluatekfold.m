function evaluatekfold(k, nbcomps, modelfolder, imgfolder, outfolder, usenms)
% Runs 'runevaluation' for each fold in a k-fold cross validation
% thing.

% Arguments
%     modelfilenames cell array of model filenames.
%     imgfolder      root folder for the folds.
%     outfolder      output folder to write results to.
%     usenms         use non-maximum suppression or not.

% Load models

for nbcomp=nbcomps
    for i=1:k
        filename = [modelfolder '/' int2str(k) '-fold-' int2str(nbcomp) ...
                    '-comps-' int2str(i) '-test-bbpred.mat'];
        model = load(filename, '-mat');
        cellmodels{i} = model.model;
    end
    
    for i = 1:k
        subfolder = [outfolder '/' int2str(mod(i,k)) '/']
        mkdir(subfolder);
        runevaluation(cellmodels{i}, [imgfolder '/' int2str(mod(i,k)) ...
                            '/positives/'], subfolder, usenms, nbcomp);
    end
end
