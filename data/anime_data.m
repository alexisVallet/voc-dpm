function [pos, neg, impos] = anime_data(posfolders, bbfolder, ...
                                        negfolders)
% Loads data from specific folders using the convention for my
% anime character detection method. Can however be used for any
% other object detection tasks provided it follows the same
% conventions. Very much inspired from pascal_data.m, hence the
% copyright notice yadda yadda.

% Arguments:
%     posfolders   folders for positive examples, containing solely
%                  images. cell array of folders.
%     bbfolder     folder containing json file specifying bounding
%                  boxes for positive examples.
%     negfolders   folders containing negative example images. cell
%                  array of folders.

% Return values
%     pos          individual positive examples (one per bounding
%                  box)
%     impos        positive example images.
%     neg          negatives examples.


% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

posfiles = cell(0);
negfiles = cell(0);
% First iterate over all the positive/negative folders to build the
% full list of positive/negative files.
nbfolders = size(posfolders);
for folderidx = 1:nbfolders
    % positives...
    posfilenames = dir(fullfile([posfolders{folderidx} '/*.jpg']));
    nbfilenames = size(posfilenames);
    posfullpaths = cell(nbfilenames);
    
    for i = 1:nbfilenames
        posfullpaths{i} = [posfolders{folderidx} '/' posfilenames(i).name];
    end
    posfiles = [posfiles; posfullpaths];

    % and negatives.
    negfilenames = dir(fullfile([negfolders{folderidx} '/*.png']));
    negfullpaths = cell(size(negfilenames));
    
    for i = 1:size(negfilenames)
        negfullpaths{i} = [negfolders{folderidx} '/' negfilenames(i).name];
    end
    negfiles = [negfiles; negfullpaths];
end

% First iterates over all positive files, populating the impos and
% pos arrays by looking up the corresponding files in bbfolder.
nbfiles = size(posfiles);
dataid = 1;
% initialize the empty impos array
impos = repmat(struct('im',[],'boxes',[],'dataids',[],'sizes',[], ...
                      'flip',false), [nbfiles, 1]);
% hard to say how large pos is going to be, but at least nbfiles
pos = repmat(struct('im',[],'x1',0,'y1',0,'x2',0,'y2',0,'boxes', ...
                    [],'flip',false,'trunc',false,'dataids',0,'sizes',0), ...
             [nbfiles, 1]);

posidx = 1;
imposidx = 1;

for i = 1:nbfiles
    fullfilename = posfiles{i};
    img = imread(fullfilename);
    [imgrows imgcols channels] = size(img);
    % lookup bounding boxes in the corresponding file
    [pathstr,name,ext] = fileparts(fullfilename);
    bboxesflat = loadjson([bbfolder '/' name '_bb.json']);
    [bbrowsflat,bbcolsflat] = size(bboxesflat);
    % loadjson flattens the bounding boxes, such that the rows are
    % alternatively dr ul dr ul etc. Which is annoying as matlab
    % uses column major order internally.
    nbboxes = bbrowsflat/2;
    bboxes = zeros(nbboxes, 4);
    for bi = 1:nbboxes
        bboxes(bi,:) = [bboxesflat(2*(bi - 1) + 1,:) bboxesflat(2*(bi ...
                                                          - 1) + 2,:)];
    end
    boxsizes = zeros(nbboxes,1);
    flippedboxes = zeros(bbrowsflat/2, bbcolsflat*2);

    for bi = 1:nbboxes
        % generate a pos entry for each bounding box
        bbox = bboxes(bi,:);
        pos(posidx).im = fullfilename;
        pos(posidx).x1 = bbox(1);
        pos(posidx).y1 = bbox(2);
        pos(posidx).x2 = bbox(3);
        pos(posidx).y2 = bbox(4);
        pos(posidx).boxes = bbox;
        pos(posidx).flip = false;
        pos(posidx).trunc = false;
        pos(posidx).dataids = dataid;
        boxsizes(bi) = (bbox(3) - bbox(1) + 1)*(bbox(4) - bbox(2) ...
                                                + 1);
        pos(posidx).sizes = boxsizes(bi);
        dataid = dataid + 1;
        posidx = posidx + 1;

        % as well as a flipped example (apparently just flip the
        % bounding box and add flip=true ?)
        flippedboxes(bi,:) = [imgcols - bbox(3) + 1, bbox(2), imgcols - bbox(1) ...
                            + 1, bbox(4)];
        pos(posidx) = pos(posidx - 1);
        pos(posidx).boxes = flippedboxes(bi,:);
        pos(posidx).x1 = flippedboxes(bi,1);
        pos(posidx).y1 = flippedboxes(bi,2);
        pos(posidx).x2 = flippedboxes(bi,3);
        pos(posidx).y2 = flippedboxes(bi,4);
        pos(posidx).dataids = dataid;
        dataid = dataid + 1;
        posidx = posidx + 1;
    end
    
    % Enter the full image info into impos
    impos(imposidx).im = fullfilename;
    impos(imposidx).boxes = bboxes;
    impos(imposidx).dataids = (dataid:dataid+nbboxes-1)';
    impos(imposidx).sizes = boxsizes;
    impos(imposidx).flip = false;
    imposidx = imposidx + 1;
    dataid = dataid + nbboxes;

    % And the flipped version
    impos(imposidx) = impos(imposidx - 1);
    impos(imposidx).boxes = flippedboxes;
    impos(imposidx).dataids = (dataid:dataid+nbboxes-1)';
    impos(imposidx).flip = true;
    imposidx = imposidx + 1;
    dataid = dataid + nbboxes;
end

% Then load negative examples in a similar fashion
nbnegfiles = size(negfiles);
neg = repmat(struct('im', [], 'flip', false, 'dataid', 0), [nbnegfiles, ...
                    1]);

for i = 1:nbnegfiles
    fullfilename = negfiles{i};
    neg(i).im = fullfilename;
    neg(i).dataid = dataid;
    dataid = dataid + 1;
end