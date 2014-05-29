function runmodel(model, folder)

files = dir(fullfile([folder '/*.jpg']));
nbfiles = size(files);

for i = 1:nbfiles
    fprintf('processing %s', files(i).name);
    fullfilename = [folder '/' files(i).name];
    img = imread(fullfilename);
    [bbox,parts] = process(img, model, -0.5);
    close all;
    showboxes(img, parts);
    waitforbuttonpress;
end