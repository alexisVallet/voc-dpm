function plotprecisionrecall(precisions, recalls)

[nbcomps, nbpoints, k] = size(precisions);

% Plot the average for each number of component
meanprecisions = zeros(nbpoints, length(nbcomps));
meanrecalls = zeros(nbpoints, length(nbcomps));
legends = cell(nbcomps,1);

for nbcomp=1:nbcomps
    precision = reshape(precisions(nbcomp,:,:), nbpoints, ...
                        k);
    recall = reshape(recalls(nbcomp,:,:), nbpoints, k);
    meanprecisions(:,nbcomp) = mean(precision,2);
    meanrecalls(:,nbcomp) = mean(recall,2);
    suffix = '';
    if nbcomp==1
        suffix = ' component';
    else
        suffix = ' components';
    end
    legends{nbcomp} = [int2str(nbcomp) suffix];
end

plot(meanrecalls, meanprecisions, 'Marker', 'x');
legend(legends);
xlabel('average recall');
ylabel('average precision');

% For each number of component, plot the average/min/max plot
% across folds.
for nbcomp=1:nbcomps
    % already computed mean precision and recall
    meanprecision = meanprecisions(:,nbcomp);
    meanrecall = meanrecalls(:,nbcomp);
    precision = reshape(precisions(nbcomp,:,:), nbpoints, ...
                        k);
    recall = reshape(recalls(nbcomp,:,:), nbpoints, k);
    figure;
    plot([meanrecall,min(recall,[],2),max(recall,[],2)], ...
         [meanprecision,min(precision,[],2),max(precision,[],2)], ...
         'Marker', 'x');
    xlabel('recall');
    ylabel('precision');
    suffix = '';
    if nbcomp==1
        suffix = ' component';
    else
        suffix = ' components';
    end
    title(['precision/recall curve for ' int2str(nbcomp) suffix]);
    legend('average', 'minimum', 'maximum');
end