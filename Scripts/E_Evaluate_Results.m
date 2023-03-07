% ----------------------------------------------------------------------------------
% "Prediction of missing eye tracking data with neural networks"
% ----------------------------------------------------------------------------------
% Lucas D. Haberkamp{1,2,3}, Michael D. Reddix{1}
% {1} Naval Medical Research Unit - Dayton
% {2} Oak Ridge Institute for Science and Education
% {3} Leidos
% ----------------------------------------------------------------------------------

%% Housekeeping
clearvars; close all; clc;

%% Load Data
% Read in a table which contains the start & stop indices for phases of flight
phaseT = readtable('../Data/Phase Start & Stop Times.csv');

yreg_groundtruth_dir = '../Data/Prep/all/ally-regression'; % Set the path to the folder storing the ground truth regression output data
yreg_groundtruth_files = dir(fullfile(yreg_groundtruth_dir, '*.csv')); % Get all files for ground truth regression output data

yclass_groundtruth_dir = '../Data/Prep/all/ally-classification'; % Set the path to the folder storing the ground truth classification output data
yclass_groundtruth_files = dir(fullfile(yclass_groundtruth_dir, '*.csv')); % Get all files for ground truth classification output data

prediction_dir = '../Data/Predictions/'; % Set the path to the folder storing the predicted data

prediction_files = dir(fullfile(prediction_dir, '*.csv')); % Get all file information in the predicted data folder

overallsummary = []; % initialize an empty array to store results for each trial

% Loop through each of the files to perform analysis
for k = 1:length(prediction_files)

    disp(prediction_files(k).name)

    filename = prediction_files(k).name(1:end-14); % extract the filename

    % Load in .csv which contains the frame number info for each file
    frames = readtable(['../Data/Frame-Numbers/',filename,'_frames.csv']);
    frames = table2array(frames);

    % determine start & stop indices for each individual file

    m = (k-1)*10; % counter for gathering start-stop indices
    idx_start = [phaseT.Var4(1+m) phaseT.Var4(3+m) phaseT.Var4(5+m) phaseT.Var4(7+m) phaseT.Var4(9+m)]; % extract start indices for 5 phases of flight
    idx_end = [phaseT.Var4(2+m) phaseT.Var4(4+m) phaseT.Var4(6+m) phaseT.Var4(8+m) phaseT.Var4(10+m)]; % extract stop indices for 5 phases of flight

    % gather list of indices which were part of the flight simulations for
    % each participant
    valid_idx = [];
    for j=1:length(idx_start)
        adj_start = find(idx_start(j)==frames);
        adj_end = find(idx_end(j)==frames);
        valid_idx = [valid_idx, adj_start:adj_end];
    end

    % Load the known regression data
    y_reg_table = readtable(fullfile(yreg_groundtruth_dir,yreg_groundtruth_files(k).name)); % reads the ground-truth regression data
    y_reg_data = table2array(y_reg_table);
    raw_CWI = y_reg_data(valid_idx,:);

    % Load in known labelled coordinates
    y_class_table = readtable(fullfile(yclass_groundtruth_dir,yclass_groundtruth_files(k).name)); % reads the ground-truth classification data
    truth_label = table2array(y_class_table);
    truth_label = truth_label(valid_idx,:);

    % Extract the neural net predictions
    ml_table = readtable(fullfile(prediction_dir,prediction_files(k).name)); % reads predicted data by the ML model
    ml_data = table2array(ml_table); % Convert the predicted data table into a array
    ml_data = ml_data(valid_idx,:); % Data which pertains to data collected during flight simulations
    pred_CWI = ml_data(:,1:3); % Predicted CWI coordinates

    pred_gaze_label = ml_data(:,4); % Extract the predicted label

    %%%%% Code to remove outliers %%%%%%
    % Data which lies in the region between the known instrument panel (+kneeboard) and out-the-window z-coordinate's (fwd-bwd) will be
    % excluded
    for i=11:13

        pred_a = nan(length(raw_CWI),1);
        if i == 11 % hdd label
            raw_a = raw_CWI(truth_label <= i, 3);
            pred_a(pred_gaze_label <= i) = pred_CWI(pred_gaze_label<=i, 3);
        else % kneeboard or out-the-window label
            raw_a = raw_CWI(truth_label==i, 3);
            pred_a(pred_gaze_label==i) = pred_CWI(pred_gaze_label==i, 3);
        end

        % add some tolerance to max & min cutoffs so that nearby and
        % reasonable predictions aren't excluded
        tolerance = 1.1; 
        cutoff_high = max(raw_a);
        if cutoff_high < 0
            cutoff_high = -1 * (abs(cutoff_high) * tolerance);
        else
            cutoff_high = cutoff_high * tolerance;
        end
        cutoff_low = min(raw_a);
        if cutoff_low < 0
            cutoff_low = -1 * (abs(cutoff_low) / tolerance);
        else
            cutoff_low = cutoff_low / tolerance;
        end

        % find indices where predictions fell outside cutoffs
        outlier_idx = pred_a > cutoff_high | pred_a < cutoff_low;

        % set invalid predictions to nan
        pred_gaze_label(outlier_idx) = nan;
    end

    % Generate a 3D plot of the predicted 3D plots with predicted label
    fig = figure('units','normalized','outerposition',[0 0 1 1]);
    for i=0:13
        if i == 11
            set(gca,'ColorOrderIndex',7);
        end
        currCWI = nan(size(pred_CWI));
        currCWI(pred_gaze_label==i,:) = pred_CWI(pred_gaze_label==i,:);
        scatter3(currCWI(:,1),-currCWI(:,3),currCWI(:,2),'filled');
        hold on
    end
    legend('Accelerometer','Airspeed', 'Attitude', 'Horizontal Situation Indicator', 'Altimeter', 'Vertical Speed', ...
        'Alternates', 'Standby', 'Engine', 'Parking', 'Flap Position', 'Misc HDD', 'Kneeboard', 'Out-the-Window');
    title([filename,': Predicted CWI & Labels']);
    xlabel('X (m)'); ylabel('Z (m)'); zlabel('Y (m)');
%   % Uncomment the line below to save the plots
%     saveas(fig,['../Plots/',filename,'_plot.png']);

    % Evaluate the predictions
    stats.total_frames = length(truth_label); % Obtain the length of the data

    stats.missing_percent.raw = sum(isnan(truth_label))/stats.total_frames*100;  % Percent missing for raw global classifications
    stats.missing_percent.pred = sum(isnan(pred_gaze_label))/stats.total_frames*100; % Percent missing for predicted global classifications

    % Get the percent of data which matched the known labels
    classification_diff = abs(pred_gaze_label - truth_label);
    classification_diff(classification_diff > 0) = 1;
    number_mislabel = nansum(classification_diff);
    stats.classification_acc = 100 - number_mislabel/sum(~isnan(classification_diff))*100;

    % Get the mean absolute error in mm
    stats.CWI_MAE = mean(nanmean(abs(pred_CWI - raw_CWI)))*1000;

    % Make a summary table for the metrics
    summaryCols = {'Subject', 'Total Frames', 'Classification Accuracy (%)', 'CWI MAE (mm)', ...
        'Raw Percent Missing (%)', 'Predicted Percent Missing (%)'};

    summarystats = [{filename} stats.total_frames stats.classification_acc stats.CWI_MAE ...
        stats.missing_percent.raw stats.missing_percent.pred];

    summaryTable = array2table(summarystats, 'VariableNames', summaryCols);
    overallsummary = [overallsummary; summaryTable]; % append the subject summary table to a master table

end

writetable(overallsummary, "../Summary Stats.csv");




