function [tp,fp,fn,matching] = evaluate_detection(gtboxes, detboxes)
% Compute the number of true positives, false positives, true
% negatives and false negative boxes for a single image. Uses the
% same method as the pascal competition. A predicted bounding box
% is considered correct if it overlaps more than 50 percent with a
% ground truth bounding box. Unlike other object detection
% benchmarks, we use a maximum bipartite matching algorithm to
% determine the exact number.

% Arguments
%     gtboxes    ground truth bounding boxes as matrix rows.
%     testboxes  boxes predicted by object detection as matrix
%                rows.

% Return values
%     [tp,fp,fn] the numbers of true/false positives and false
%                negatives. There isn't really a notion of true
%                negatives here.

% First build the bipartite graph for overlappings by testing all
% pairs.
    [nbgt pouet] = size(gtboxes);
    [nbdet prout] = size(detboxes);

    overlap_graph = graph(nbgt + nbdet);

    gtpart = 1:nbgt;
    detpart = nbgt+1:nbgt+nbdet;

    for i = 1:nbgt
        for j = 1:nbdet
            % If the boxes overlap more than 50% consider a
            % candidate detection.
            interarea = boxarea(intersection(gtboxes(i,:), ...
                                             detboxes(j,:)));
            unionarea = boxarea(gtboxes(i,:)) + ...
                boxarea(detboxes(j,:)) - interarea;
            
            if (interarea/unionarea) >= 0.5
                add(overlap_graph, gtpart(i), detpart(j));
            end
        end
    end
    
    % call the maximum matching algorithm, and we're done a couple
    % of subtraction later.
    matching = bipmatch(overlap_graph, gtpart, detpart);
    % bring back indexes for detected boxes in the original range
    matching(:,2) = matching(:,2) - nbgt;
    % The number of true positives is precisely the cardinality of
    % a maximum matching in the overlapping graph.
    [tp bidule] = size(matching);
    % The number of false negatives is the number of unconnected
    % ground truth boxes by the matching. Same reasoning for false
    % positives and detected boxes.
    fp = nbdet - tp;
    fn = nbgt - tp;
end

function area = boxarea(bbox)
    if bbox(3) > bbox(1) && bbox(4) > bbox(1)
        area = (bbox(3) - bbox(1) + 1) * (bbox(4) - bbox(2) + 1);
    else
        area = 0;
    end
end

function bbox = intersection(bbox1, bbox2)
    bbox = [max(bbox1(1), bbox2(1)), max(bbox1(2), bbox2(2)), ...
            min(bbox1(3), bbox2(3)), min(bbox1(4), bbox2(4))];
end
