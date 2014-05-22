function [pos,neg,impos] = load_data(folder)
% Get training data from an arbitrary dataset.

% Return values
%   pos     Each positive example on its own
%   neg     Each negative image on its own
%   impos   Each positive image with a list of foreground boxes

% Arguments
%   folder  Folder containing images and corresponding bounding box
%           data in {imgname}_bb.json

