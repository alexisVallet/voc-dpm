function evaluatekfold(models, imgfolder, outfolder, usenms)
% Runs 'runevaluation' for each fold in a k-fold cross validation
% thing.

% Arguments
%     models    struct array of DPMs.
%     imgfolder root folder for the folds.
%     outfolder output folder to write results to.
%     usenms    use non-maximum suppression or not.

if argin < 4
    usenms = false;
end

k = len(models);

for i = 1:k
    subfolder = [outfolder '/' int2str(i) '/'];
    mkdir(subfolder);
    runevaluation(model, [imgfolder '/' int2str(i) '/positives/'], ...
                  subfolder, usenms);
end