function models = train_bboxpred_kfold(k, models, imgfolder, gtfolder)
% Trains a bounding box predictor for all folds of a k-fold cross
% validation.

% Arguments
%     k         the number of folds
%     models    struct array of models to train for bbox prediction,
%               from first to kth fold as specified by the directory
%               structure of imgfolder.
%     imgfolder folder containing the folds for the cross
%               validation.
%     gtfolder  folder containing ground truth bounding boxes.

% Return value
%     models    the same array as the input models, with bounding
%               box predictors trained for each model.

% Get subfolders from the image folder
posfolders = cell(k,1);
negfolders = cell(k,1);

for i = 1:k
    % Alright so the reason why it's mod(i,5) is that I messed up
    % the indices at first (the folds are stored in 0 based index)
    % but only noticed in the middle of training (which takes a
    % long time). To save the already cached results, I just
    % swapped 5 for 0.
    posfolders{i} = [imgfolder '/' int2str(mod(i, 5)) '/positives'];
    negfolders{i} = [imgfolder '/' int2str(mod(i, 5)) '/negatives'];
end

tmpmodels = cell(k, 1);

% Train all the models for bounding box prediction
for i=1:k
    % find out which folders are training data for this fold
    trainidxs = [1:i-1 i+1:k];
    trainpos = posfolders(trainidxs);
    trainneg = posfolders(trainidxs);

    tmpmodels{i} = general_bboxpred_train(models(i), trainpos, ...
                                          gtfolder, trainneg);
end

% Because matlab sucks balls
models = cell2mat(tmpmodels);

