function runevaluation(model, testfolder, outfolder, usenms)
% Rune image detection on each image of the test folder using the
% specified model, and writes the resulting bounding boxes (with
% associated score) in the output folder. This includes detection,
% bounding box prediction (if available), clipping and optionally
% non-maximum suppression.

% defaults to false. Nms is cool for pascal as it seems there are no
% overlapping boxes in the dataset. That's not necessarily the case
% in general.
if nargin < 4
    usenms = false;
end

% for each image in the test folder, run the whole shebang
imgfiles = dir(fullfile([testfolder '/*.jpg']));
nbfiles = size(imgfiles);

if not(isfield(model, 'bboxpred'))
    fprintf(['The model hasnt been trained for bounding box ' ...
             'prediction!']);
end

for i = 1:nbfiles
    % run detection with low threshold, bounding box prediction,
    % clipping and optionally nms, before saving the resulting
    % boxes and score to output.
    img = imread([testfolder '/' imgfiles(i).name]);
    [ds bs trees] = imgdetect(img, model, model.thresh);
    % clipping
    [ds_clip bs_clip] = clipboxes(img, ds, bs);
    % bounding box prediction if available
    ds_pred = [];
    bs_pred = [];
    if isfield(model, 'bboxpred')
        [ds_pred bs_pred] = bboxpred_get(model.bboxpred, ds_clip, ...
                                         reduceboxes(model, bs_clip));
    else
        ds_pred = ds
        bs_pred = bs
    end
    % optional non maximum suppression
    ds_nms = [];
    if usenms
        I = nms(ds_pred, 0.5);
        ds_nms = ds_pred(I,:);
    else
        ds_nms = ds_pred;
    end

    % save the resulting ds array
    [pathstr, stem] = fileparts(imgfiles(i).name);
    save([outfolder '/' stem '_detbb.mat'], 'ds_nms');
end
