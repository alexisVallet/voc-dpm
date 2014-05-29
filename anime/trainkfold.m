function trainkfold(nbcomps, k, imgfolder, bbfolder, outfolder)
% Trains k  deformable parts model given k datasets, training all
% possible k-1 combinations of datasets for the purposes of k-fold
% cross validation, writing models to an output folder.

% Arguments
%     nbcomps    vector of number of components for the deformable
%                part models.
%     k          number of folds.
%     imgfolder  folder for images.
%     bbfolder   folder for bounding boxes for all positive
%                examples.
%     outfolder  folder to write trained models to.

% Get subfolders from the image folder
posfolders = cell(k,1);
negfolders = cell(k,1);

for i = 1:k
    posfolders{i} = [imgfolder '/' int2str(i) '/positives'];
    negfolders{i} = [imgfolder '/' int2str(i) '/negatives'];
end

% Iterate over all number of components and test sets
for nbcomp = nbcomps
    for testsetidx = 1:k
        % generate data for the corresponding training sets
        trainingidxs = [1:testsetidx-1 testsetidx+1:k];
        trainingposfolders = posfolders(trainingidxs);
        trainingnegfolders = negfolders(trainingidxs);
        
        [pos,neg,impos] = anime_data(trainingposfolders, bbfolder, ...
                                     trainingnegfolders, nbcomp, ...
                                     '');
        % run the actual training
        name = [int2str(k) '-fold-' int2str(nbcomp) '-comps-' ...
                int2str(testsetidx) '-test'];
        model = general_train(pos,impos,neg,nbcomp,'',name);
        save([outfolder '/' name '.mat'], 'model');
    end
end

     