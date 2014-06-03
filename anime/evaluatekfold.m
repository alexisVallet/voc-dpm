function evaluatekfold(modelfilenames, imgfolder, outfolder, usenms)
% Runs 'runevaluation' for each fold in a k-fold cross validation
% thing.

% Arguments
%     modelfilenames cell array of model filenames.
%     imgfolder      root folder for the folds.
%     outfolder      output folder to write results to.
%     usenms         use non-maximum suppression or not.

if nargin < 4
    usenms = true;
end

% Load models
k = size(modelfilenames);
k = k(1);
cellmodels = cell(k,1);

for i=1:k
    filename = modelfilenames{mod(i,k) + 1};
    model = load(filename, '-mat');
    cellmodels{i} = model.model;
end

for i = 1:k
    subfolder = [outfolder '/' int2str(mod(i,k)) '/']
    mkdir(subfolder);
    runevaluation(cellmodels{i}, [imgfolder '/' int2str(mod(i,k)) '/positives/'], ...
                  subfolder, usenms);

end
