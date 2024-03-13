clc;clear;close all;

root_path = pwd; 
toolbox = fullfile(pwd,'matlab-toolbox');
addpath(genpath(toolbox));
data_path = fullfile(root_path, 'data', filesep);
fulldata_path = fullfile(data_path,'MHK_BR_10'); % -----------------Tunable

%% Options for different simulation results
% prefix and suffix of output files

prefix = 'RM1_'; % simulations without TTDspFA
suffix = '.outb';

%% Load OpenFAST simulations
% get all the files in the directory
Out_files = dir(fullfile(fulldata_path,[prefix,'*',suffix]));

tmin = 100;

% get length
nLinCases = length(Out_files);
 %nLinCases = 1;

% numbering scheme bases on the number of files
if nLinCases <= 10
    numstring = '%01d';
else
    numstring = '%02d';
end

% Required states
req_states = {'PtfmPitch','GenSpeed'};

% required controls
req_controls = {'Wind1VelX','GenTq','BldPitch1','Wave1Elev'};


% go through each case and load OpenFAST simulations
% The variable channels stores the simulations results.
% If there are 200 channels, and there are 2000 points in time, then the
% size of channels is [2000,200].
% The first column is always Time

% index for time
iTime = 1;


for iCase = 1

    % Read the binary file and load the simulation results
    [Channels, ChanName, ChanUnit, DescStr] = ReadFASTbinary(fullfile(fulldata_path,[prefix,num2str(iCase-1,numstring),suffix]));
      

    % Go thorugh the list of 'req_states' and extract the indices for the states
    for i = 1:length(req_states)
        iStates(i) = find(contains(ChanName,req_states{i}));
    end

    % Go thorugh the list of 'req_controls' and extract the indices for the controls
    for i = 1:length(req_controls)
        iInputs(i) = find(contains(ChanName,req_controls{i}));
    end

    % extract
    time = Channels(:,iTime);
    t_ind = time > tmin; % ignore the first tmin seconds of the simulation
    time = time(t_ind);

    % extract states and controls
    states = Channels(t_ind,iStates);
    controls = Channels(t_ind,iInputs);

   
end

% names
state_names = req_states;

% Input and Output for the neural network model
Input = controls';
Output = states';


% plot the inputs and outputs
hf = figure;
hf.Color = 'w';

subplot(5,1,1)
plot(time,controls(:,1))
title('Wind Speed')

subplot(5,1,2)
plot(time,controls(:,2))
title('GenTq')

subplot(5,1,3)
plot(time,controls(:,3))
title('BldPitch')

subplot(5,1,4)
plot(time,states(:,1))
title('PtfmPitch')

subplot(5,1,5)
plot(time,states(:,2))
title('GenSpeed')


%% Preprocssing data
% split data into training and testing

nt = length(time);


nsplit = 50; %<--------------------------- Tunable.
% This parameter corresponds to the percentage of data we use for traning
% and testing

%We split the nt number of data into traning and testing

% index for training data
indTrain = 1: floor(nsplit/100*nt);
ntrain = length(indTrain);

% index for testing data
indTest = indTrain(end)+1:nt;
ntest = length(indTest);

% extract the testing and training data
XTrainA = Input(:,indTrain);
TTrainA = Output(:,indTrain);

XTestA = Input(:,indTest);
TTestA = Output(:,indTest);

time_train = time(indTrain); 
time_test = time(indTest);

n_inputs = size(Input,1);
n_outputs = size(Output,1);

% calculate the mean and standard deviation, and
% normalize the data

muX = mean(XTrainA,2);
sigmaX = std(XTrainA,0,2);

muT = mean(TTrainA,2);
sigmaT = std(TTrainA,0,2);

% Reshape training data into 100 differet cells
% reshape into cell array
ncells = 100;

% get index
indTrain_ = reshape(indTrain,[],ncells);

% initialize
XTrain = cell(1,ncells);
TTrain = cell(1,ncells);

% Reshape and normalize
for i = 1:ncells
    ind_ = indTrain_(:,i);

    XTrain{i} = (XTrainA(:,ind_) - muX)./sigmaX;
    TTrain{i} = (TTrainA(:,ind_) - muT)./sigmaT;

end

%% Train LSTM

% define architecture
layers = [
    sequenceInputLayer(n_inputs)
    lstmLayer(128)
    fullyConnectedLayer(n_outputs)
    regressionLayer];

% training options
max_epoch = 200; %--------------------------- Tunable

options = trainingOptions("adam", ...
    MaxEpochs=max_epoch, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=1);

% train the LSTM network

tic;
net = trainNetwork(XTrain,TTrain,layers,options);
t_train = toc;

% normalize test
XTest = (XTestA - muX)./sigmaX;
TTest = (TTestA - muT)./sigmaT;

%% Test LSTM

% initalize storage array
states_lstm = zeros(n_outputs,ntest);

% predict using the trained net
tic;
for i = 1:ntest

    % extract
    input_ = XTest(:,i);

    % predict
    [net,states_lstm(:,i)] = predictAndUpdateState(net,input_);


end
t_predict = toc;

% renormalize
states_lstm = states_lstm.*sigmaT;
states_lstm = states_lstm + muT;

states_act = TTest.*sigmaT;
states_act = states_act + muT;


% plot results from LSTM
hf = figure;
hf.Color = 'w';

ind = 1;

subplot(2,1,ind)
hold on;
plot(time_test,states_act(ind,:),'k')
plot(time_test,states_lstm(ind,:),'r')
xlabel('Time'); ylabel('PtfmPitch')
legend('original','LSTM')


ind = ind + 1;

subplot(2,1,ind)
hold on;
plot(time_test,states_act(ind,:),'k')
plot(time_test,states_lstm(ind,:),'r')
xlabel('Time'); ylabel('GenSpeed')
legend('original','LSTM')

return
