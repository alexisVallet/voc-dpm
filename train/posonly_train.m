function model = posonly_train(pos, impos, neg, n, note)
% Train a model given positive and negative examples. Code adapted
% from pascal_train to be more flexible for our purposes.

% Arguments
%     pos    Each positive example on its own
%     impos  Each positive image with a list of foreground boxes
%     neg    Each negative example
%     n      Number of components for the deformable parts model
%     note   A note to attach to the output model

% Return values
%     model  A model trained to detect the objects specified in
%            positive examples

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

% We'll assume there is only one class, name it with a placeholder
cls = 'onlyclass';

seed_rand();

% Default to no note
if nargin < 3
  note = '';
end

% Still load the config from voc_config(). We'll be ignoring
% anything pascal specific though.
conf = voc_config();
cachedir = conf.paths.model_dir;

% Split foreground examples into n groups by aspect ratio
spos = split(pos, n);

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Select a small, random subset of negative images
% All data mining iterations use this subset, except in a final
% round of data mining where the model is exposed to all negative
% images
num_neg   = length(neg);
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));
neg_large = neg; % use all of the negative images

% Train one asymmetric root filter for each aspect ratio group
% using warped positives and random negatives
try
  load([cachedir cls '_lrsplit1']);
catch
  seed_rand();
  for i = 1:n
    models{i} = root_model(cls, spos{i}, note);
    % Split the i-th aspect ratio group into two clusters: 
    % left vs. right facing instances
    inds = lrsplit(models{i}, spos{i});
    % Train asymmetric root filter on one of these groups
    models{i} = train(models{i}, spos{i}(inds), neg_large, true, true, 1, 1, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit1_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit1'], 'models');
end

% Train a mixture of two root filters for each aspect ratio group
% Each pair of root filters are mirror images of each other
% and correspond to two latent orientations choices
% Training uses latent positives and hard negatives
try
  load([cachedir cls '_lrsplit2']);
catch
  seed_rand();
  for i = 1:n
    % Build a mixture of two (mirrored) root filters
    models{i} = lr_root_model(models{i});
    models{i} = train(models{i}, spos{i}, neg_small, false, false, 4, 3, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit2_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit2'], 'models');
end

% Train a mixture model composed all of aspect ratio groups and 
% latent orientation choices using latent positives and hard negatives
try 
  load([cachedir cls '_mix']);
catch
  seed_rand();
  % Combine separate mixture models into one mixture model
  model = model_merge(models);
  model = train(model, impos, neg_small, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, false, 'mix');
  save([cachedir cls '_mix'], 'model');
end

% Train a mixture model with 2x resolution parts using latent positives
% and hard negatives
try 
  load([cachedir cls '_parts']);
catch
  seed_rand();
  % Add parts to each mixture component
  for i = 1:2:2*n
    % Top-level rule for this component
    ruleind = i;
    % Top-level rule for this component's mirror image
    partner = i+1;
    % Filter to interoplate parts from
    filterind = i;
    model = model_add_parts(model, model.start, ruleind, ...
                            partner, filterind, 8, [6 6], 1);
    % Enable learning location/scale prior
    bl = model.rules{model.start}(i).loc.blocklabel;
    model.blocks(bl).w(:)     = 0;
    model.blocks(bl).learn    = 1;
    model.blocks(bl).reg_mult = 1;
  end
  % Train using several rounds of positive latent relabeling
  % and data mining on the small set of negative images
  model = train(model, impos, neg_small, false, false, 8, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'parts_1');
  % Finish training by data mining on all of the negative images
  model = train(model, impos, neg_large, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'parts_2');
  save([cachedir cls '_parts'], 'model');
end

save([cachedir cls '_final'], 'model');
