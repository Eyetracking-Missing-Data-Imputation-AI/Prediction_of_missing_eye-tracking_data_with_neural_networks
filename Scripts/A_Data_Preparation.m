% ----------------------------------------------------------------------------------
% "Prediction of missing eye tracking data with neural networks"
% ----------------------------------------------------------------------------------
% Lucas D. Haberkamp{1,2,3}, Michael D. Reddix{1}
% {1} Naval Medical Research Unit - Dayton
% {2} Oak Ridge Institute for Science and Education
% {3} Leidos
% ----------------------------------------------------------------------------------

%% Housekeeping
clc; clearvars; close all; 

%% Load data & obtain file information
filepaths.raw = '../Data/Raw/';  % Set the path to the folder storing the raw data

myFiles = dir(fullfile(filepaths.raw, '*.csv')); % Get all file information in the raw data folder

%% Specify file output paths

filepaths.train.x = '../Data/Prep/train/trainx/'; % trainx (inputs to the model) filepath

filepaths.train.y_reg = '../Data/Prep/train/trainy-regression/'; % trainy (regression model outputs) filepath

filepaths.train.y_class = '../Data/Prep/train/trainy-classification/'; % labely (classification model outputs) filepath

filepaths.validation.x = '../Data/Prep/validation/valx/'; % valx (inputs to the model) filepath

filepaths.validation.y_reg = '../Data/Prep/validation/valy-regression/'; % valy (regression model outputs) filepath

filepaths.validation.y_class = '../Data/Prep/validation/valy-classification/'; % labely (classification model outputs) filepath

filepaths.all.x = '../Data/Prep/all/allx/'; % allx (inputs to the model after model training) filepath

filepaths.all.y_reg = '../Data/Prep/all/ally-regression/'; % ally (regression model outputs) filepath

filepaths.all.y_class = '../Data/Prep/all/ally-classification/'; % labely (classification model outputs) filepath


%% Randomly select participants to be put into the validation dataset
validation_set_size = 2; % Specify number of files to be placed in the validation set

% Randomly select indexes of files to be put in the validation set
rng(1); % set seed
val_set_idx = randperm(length(myFiles),validation_set_size); % randomly select the participants to withhold

fprintf("Participants in the validation set are P%d & P%d\n",val_set_idx(1),val_set_idx(2))

% Loop through files in the directory to perform pre-processing
for k = 1:length(myFiles)

    % Import the raw data for the current participant
    filename = myFiles(k).name;
    disp(myFiles(k).name)

    opts = detectImportOptions(fullfile(filepaths.raw,filename));
    data = readtable(fullfile(filepaths.raw,filename), opts);
    f = filename(1:end-4);

    % Select the variables to be used as inputs to the model
    xCols =  {'LeftGazeDirection_x', 'LeftGazeDirection_y','LeftGazeDirection_z',...
        'RightGazeDirection_x', 'RightGazeDirection_y', 'RightGazeDirection_z',...
        'LeftGazeOrigin_x', 'LeftGazeOrigin_y', 'LeftGazeOrigin_z',...
        'RightGazeOrigin_x', 'RightGazeOrigin_y', 'RightGazeOrigin_z', ...
        'LeftEyelidOpening', 'RightEyelidOpening'};
    
    x_table = data(:,xCols); % extract x inputs from the raw data table 
        
    allx_file = [filepaths.all.x,f,'_allx.csv']; % add the filename to the filepath for export
    writetable(x_table, allx_file); % export the data to a .csv file

    % prepare the training input data
    if k~=val_set_idx % participants not in the validation dataset
        trainx_file = [filepaths.train.x ,f, '_trainx.csv']; % add the filename to the filepath for export
        writetable(x_table, trainx_file); % export the training data to a .csv file
    else
        valx_file = [filepaths.validation.x, f, '_valx.csv']; % export the unmodified x data for validation files
        writetable(x_table, valx_file);
    end
    
    % extract point of gaze coordinate data 
    yCols = {'ClosestWorldIntersection_worldPoint_x',...
        'ClosestWorldIntersection_worldPoint_y',...
        'ClosestWorldIntersection_worldPoint_z'};
        
    yData = table2array(data(:,yCols));
    yData(yData==0) = nan; % zeros are missing data / set to nan
    
    label_column = data.ClosestWorldIntersection_objectName; % extract label names for existing coordinate labels       
        
    label = nan(length(label_column),1); % Initiate a empty variable for encoding appropriate label names
    
    % Find all existing coordinates which include an HDD label
    hdd_pts = label; 
    hdd_pts(contains(label_column, 'HDD')) = 1;
    
    % Encode relevant labels
    % 0 = Accelerometer
    label(contains(label_column, 'Accelerometer')) = 0;
    % 1 = Airspeed
    label(contains(label_column, 'Airspeed')) = 1;
    % 2 = Attitude
    label(contains(label_column, 'Attitude')) = 2;
    label(strcmp(label_column,'HDD.Slip Turn Indicator')) = 2; 
    % 3 = HSI
    label(contains(label_column, 'Horizontal Situation Indicator')) = 3;
    % 4 = Altimeter
    label(contains(label_column, 'Altimeter')) = 4;
    % 5 = Vertical Speed
    label(contains(label_column, 'Vertical')) = 5;
    % 6 = Alternates
    label(contains(label_column, 'Alternate')) = 6;
    % 7 = Standby
    label(contains(label_column, 'Standby')) = 7;
    % 8 = Engine
    label(strcmp(label_column,'HDD.Engine Torque')) = 8;
    label(strcmp(label_column,'HDD.Primary Engine Data Display Panel')) = 8;
    label(strcmp(label_column,'HDD.N1')) = 8;
    label(strcmp(label_column,'HDD.NP & 10AT')) = 8;
    label(strcmp(label_column,'HDD.ITT')) = 8;
    % 9 = Parking Brake
    label(strcmp(label_column,'HDD.Parking Brake Handle')) = 9;
    % 10 = Landing Gear
    label(strcmp(label_column,'HDD.Flap Position Indicator')) = 10;
    label(strcmp(label_column,'HDD.Landing Gear Control Panel')) = 10;

    % Set labels that aren't part of the HDD to nan
    hdd_pts(~isnan(label)) = nan;
    
    % get xy coordinates of the CWI coordinates for data on the hdd
    hdd_coord = nan(length(yData),2);
    hdd_coord(hdd_pts==1,1:2) = yData(hdd_pts==1,1:2);
    
    % CWI coordinates on the accelerometer
    acc_coord = nan(length(yData),2);
    acc_coord(label==0,1:2) = yData(label==0,1:2);
    
    % CWI coordinates on the airspeed indicator
    air_coord = nan(length(yData),2);
    air_coord(label==1,1:2) = yData(label==1,1:2);   
    
    % CWI coordinates on the attitude indicator
    att_coord = nan(length(yData),2);
    att_coord(label==2,1:2) = yData(label==2,1:2); 
    
    % CWI coordinates on the standby indicator
    standby_coord = nan(length(yData),2);
    standby_coord(label==6,1:2) = yData(label==6,1:2);
        
    % CWI coordinates on the HSI indicator
    hsi_coord = nan(length(yData),2);
    hsi_coord(label==3,1:2) = yData(label==3,1:2);
    
    % CWI coordinates on the landing gear
    land_coord = nan(length(yData),2); 
    land_coord(label==10,1:2) = yData(label==10,1:2); 
    
    % CWI coordinates on the parking brake
    park_coord = nan(length(yData),2); 
    park_coord(label==9,1:2) = yData(label==9,1:2); 
    
    % CWI coordinates on the engine indicator
    eng_coord = nan(length(yData),2); 
    eng_coord(label==8,1:2) = yData(label==8,1:2);
    
    % CWI coordinates on the altitude indicator
    alt_coord = nan(length(yData),2); 
    alt_coord(label==4,1:2) = yData(label==4,1:2); 
    
    % Determine which coordinates correspond with a radio label by:
    % accelerometer x < x < attitude x; hsi y < y < airspeed y
    loc = hdd_coord(:,1) > max(acc_coord(:,1))*0.9 & hdd_coord(:,1) < min(att_coord(:,1))*1.1 ...
        & hdd_coord(:,2) < min(air_coord(:,2))*1.1 & hdd_coord(:,2) > min(hsi_coord(:,2))*1.1;
    
    % 11 = Radio / Miscellaneous HDD
    label(loc) = 11; 
    hdd_coord(loc,:) = nan;
    
    % CWI coordinates on the radio
    radio_coord = nan(length(yData),2);
    radio_coord(label==11,1:2) = yData(label==11,1:2);
    
    % Determine which coordinates correspond with a GPS label by:
    %  x < radio x; hsi y < y < accelerometer y
    loc = hdd_coord(:,1) < min(radio_coord(:,1))*1.2 & hdd_coord(:,2) < min(acc_coord(:,2))*1.1 ...
        & hdd_coord(:,2) > min(hsi_coord(:,2))*1.1;
    
    % 11 = GPS / Miscellaneous HDD
    label(loc) = 11; 
    hdd_coord(loc,:) = nan;    
    
    % Determine which coordinates are below known instruments on the HDD by:
    %  landing gear x < x < radio x; hsi y < y 
    loc = hdd_coord(:,1) > max(land_coord(:,1))*0.7 & hdd_coord(:,2) < min(hsi_coord(:,2))*1.3 ...
    & hdd_coord(:,1) < min(park_coord(:,1))*0.9;

    % 11 = Lower HDD Gaze / Miscellaneous HDD
    label(loc) = 11; 
    hdd_coord(loc,:) = nan;
    
    % Determine which coordinates are right of known instruments on the HDD by:
    %  x > altitude x; y > engine y
    loc = hdd_coord(:,1) > max(alt_coord(:,1))*1.3 & hdd_coord(:,2) > max(eng_coord(:,2))*0.8;
    
    % 11 = Right HDD Gaze / Miscellaneous HDD
    label(loc) = 11; 
    hdd_coord(loc,:) = nan;
    
    % Determine which coordinates are left of known instruments on the HDD by:
    %  x < accelerometer x; y > accelerometer y
    loc = hdd_coord(:,1) < min(acc_coord(:,1))*1.3 & hdd_coord(:,2) > min(acc_coord(:,2))*0.8;
    
    % 11 = Top Left HDD Gaze / Miscellaneous HDD
    label(loc) = 11; 
    hdd_coord(loc,:) = nan;
    
    % 12 = Kneeboard
    label(contains(label_column, 'KneeBoard')) = 12;
    % 13 = Out the Window Labels
    label(contains(label_column, 'BugEyeHex01')) = 13;
    label(contains(label_column, 'BugEyeHex02')) = 13;
    label(contains(label_column, 'BugEyeHex04')) = 13;
    label(contains(label_column, 'BugEyeHex05')) = 13;
    label(contains(label_column, 'BugEyeHex06')) = 13;
    label(contains(label_column, 'BugEyeHex07')) = 13;
    label(contains(label_column, 'BugEyeHex08')) = 13;
    label(contains(label_column, 'BugEyeHex09')) = 13;
    label(contains(label_column, 'BugEyeHex10')) = 13;
    
    % Raw HDD data is not flush along the HDD. This causes issues
    % in classification. Set the hdd z axis to have the same angle.
    
    % extract y-values for the hdd labels
    y_array = yData(label <= 11,2); 
    y_array = y_array(~isnan(y_array));

    % extract z-values for the hdd labels
    z_array = yData(label <= 11,3);
    z_array = z_array(~isnan(z_array));

    % find the min & max values for the y-axis and the corresponding z-axis values 
    [y2, loc] = max(y_array); z2 = z_array(loc);
    [y1, loc] = min(y_array); z1 = z_array(loc);

    % find the angle between the min & max y-axis values
    theta = atan2d((y2 - y1),(z2 - z1));
    disp(theta);

    % set the angle found between the current point and minimum y-value to
    % have the angle found between the coordinates that had the min & max
    % y-value
    for ii = 1:length(yData)
        if label(ii) <= 11
            y_curr = yData(ii,2);
            z_new = (y_curr - y1)/(tand(theta))+z1;
            yData(ii,3) = z_new;
        end
    end
    
    % set unlabelled CWI coordinates to nan
    yData(isnan(label),:) = nan;
    
    % Generate a 3D plot with the original labels
    fig = figure('units','normalized','outerposition',[0 0 1 1]);
    for i=0:13
        if i == 11
            set(gca,'ColorOrderIndex',7);
        end
        currCWI = nan(size(yData));
        currCWI(label==i,:) = yData(label==i,:);
        scatter3(currCWI(:,1),-currCWI(:,3),currCWI(:,2),'filled');
        hold on
    end
    legend('Accelerometer','Airspeed', 'Attitude', 'Horizontal Situation Indicator', 'Altimeter', 'Vertical Speed', ...
        'Alternates', 'Standby', 'Engine', 'Parking', 'Flap Position', 'Misc HDD', 'Kneeboard', 'Out-the-Window');
    title([f,': Ground Truth CWI & Labels']);
    xlabel('X (m)'); ylabel('Z (m)'); zlabel('Y (m)')
    % %Uncomment below to save the plots
    % saveas(fig,['../Plots/',filename,'_rawplot.png']); 
         
    % Place the regression (CWI) data in a table to prepare for export 
    yRegTable = array2table(yData, 'VariableNames', yCols); 
    
    % Place the labelled and encoded data in a table to prepare for export 
    ylabel_Table = array2table(label,'VariableNames',"Label"); 

    % write results to the ally-regression folder
    ally_reg_file = ([filepaths.all.y_reg, f, '_allyreg.csv']);
    writetable(yRegTable, ally_reg_file);
    
    % write results to the ally-classification folder
    ally_lab_file = ([filepaths.all.y_class,f,'_allylab.csv']); 
    writetable(ylabel_Table, ally_lab_file); 

    if k~=val_set_idx % participants not in the validation dataset
        % write results to the trainy-regression folder
        trainy_reg_file = ([filepaths.train.y_reg,f,'_trainyreg.csv']);
        writetable(yRegTable, trainy_reg_file);
        % write results to the trainy-classification folder
        trainy_lab_file = ([filepaths.train.y_class,f,'_trainylab.csv']); 
        writetable(ylabel_Table, trainy_lab_file); 
    else
        % write results to the valy-regression folder
        valy_reg_file = ([filepaths.validation.y_reg,f,'_valyreg.csv']);
        writetable(yRegTable, valy_reg_file); 
        % write results to the trainy-classification folder
        valy_lab_file = ([filepaths.validation.y_class,f,'_valylab.csv']);
        writetable(ylabel_Table, valy_lab_file); 
    end
    
end



