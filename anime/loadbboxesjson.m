function bboxes = loadbboxesjson(jsonfile)
bboxesflat = loadjson(jsonfile);
[bbrowsflat,bbcolsflat] = size(bboxesflat);
% loadjson flattens the bounding boxes, such that the rows are
% alternatively dr ul dr ul etc. Which is annoying as matlab
% uses column major order internally.
nbboxes = bbrowsflat/2;
bboxes = zeros(nbboxes, 4);
for bi = 1:nbboxes
    bboxes(bi,:) = [bboxesflat(2*(bi - 1) + 1,:) ...
                    bboxesflat(2*(bi - 1) + 2,:)];
end
