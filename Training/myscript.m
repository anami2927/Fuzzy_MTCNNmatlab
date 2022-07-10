%% step 1 Load Training Data
[XTrain,T1Train,T2Train] = digitTrain4DArrayData;

% digitTrain4DArrayData   Load the digit training set as 4-D array data
%  
%     images      - Input data as an H-by-W-by-C-by-N array, where H is the
%                   height and W is the width of the images, C is the number
%                   of channels, and N is the number of images.
%     digits      - Categorical vector containing the labels for each
%                   observation.
%     angles      - Numeric vector containing the angle of rotation in
%                   degrees for each image.


% Datastore for in-memory data
dsXTrain = arrayDatastore(XTrain,IterationDimension=4); 
dsT1Train = arrayDatastore(T1Train);
dsT2Train = arrayDatastore(T2Train);

dsTrain = combine(dsXTrain,dsT1Train,dsT2Train);

classNames = categories(T1Train);
numClasses = numel(classNames);
numResponses = size(T2Train,2);
numObservations = numel(T1Train);

%% 1.1 view some image
idx = randperm(numObservations,64);
I = imtile(XTrain(:,:,:,idx));
figure
imshow(I)
%% step 2 Define Deep Learning Model
% Define the following network that predicts both labels and angles of rotation.
% 
% A convolution-batchnorm-ReLU block with 16 5-by-5 filters.
% 
% A branch of two convolution-batchnorm blocks each with 32 3-by-3 filters with a ReLU operation between
% 
% A skip connection with a convolution-batchnorm block with 32 1-by-1 convolutions.
% 
% Combine both branches using addition followed by a ReLU operation
% 
% For the regression output, a branch with a fully connected operation of size 1 (the number of responses).
% 
% For classification output, a branch with a fully connected operation of size 10 (the number of classes) and a softmax operation.
% 

% Define and Initialize Model Parameters and State

% Define the parameters for each of the operations and include them in a struct. 
% Use the format parameters.OperationName.ParameterName where parameters is the struct, 
% OperationName is the name of the operation (for example "conv1") and 
% ParameterName is the name of the parameter (for example, "Weights").

filterSize = [5 5];
numChannels = 1;
numFilters = 16;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

%% Initialize the parameters for the first convolution operation, "conv1".
parameters.conv1.Weights = initializeGlorot(sz,numOut,numIn); % random initial weights 
parameters.conv1.Bias = initializeZeros([numFilters 1]); % initial bias with zeros

%% Initialize the parameters and state for the first batch normalization operation, "batchnorm1".
% Initialize the batch normalization offset and scale parameters with the initializeZeros and initializeOnes example functions, respectively.
% To perform training and inference using batch normalization operations, 
% you must also manage the network state. Before prediction, 
% you must specify the dataset mean and variance derived from the training data. 
% Create a structure state containing the state parameters. 
% The batch normalization statistics must not be dlarray objects. 

parameters.batchnorm1.Offset = initializeZeros([numFilters 1]);
parameters.batchnorm1.Scale = initializeOnes([numFilters 1]);


% Initialize the batch normalization trained mean and trained variance states using the zeros and ones functions, respectively.

state.batchnorm1.TrainedMean = initializeZeros([numFilters 1]);
state.batchnorm1.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the second convolution operation, "conv2".
filterSize = [3 3];
numChannels = 16;
numFilters = 32;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.conv2.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv2.Bias = initializeZeros([numFilters 1]);

%% Initialize the parameters and state for the second batch normalization operation, "batchnorm2".
parameters.batchnorm2.Offset = initializeZeros([numFilters 1]);
parameters.batchnorm2.Scale = initializeOnes([numFilters 1]);
state.batchnorm2.TrainedMean = initializeZeros([numFilters 1]);
state.batchnorm2.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the third convolution operation, "conv3".
filterSize = [3 3];
numChannels = 32;
numFilters = 32;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.conv3.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv3.Bias = initializeZeros([numFilters 1]);

%% Initialize the parameters and state for the third batch normalization operation, "batchnorm3".
parameters.batchnorm3.Offset = initializeZeros([numFilters 1]);
parameters.batchnorm3.Scale = initializeOnes([numFilters 1]);

state.batchnorm3.TrainedMean = initializeZeros([numFilters 1]);
state.batchnorm3.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the convolution operation in the skip connection, "convSkip".
filterSize = [1 1];
numChannels = 16;
numFilters = 32;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.convSkip.Weights = initializeGlorot(sz,numOut,numIn);
parameters.convSkip.Bias = initializeZeros([numFilters 1]);

%% Initialize the parameters and state for the batch normalization operation in the skip connection, "batchnormSkip".
parameters.batchnormSkip.Offset = initializeZeros([numFilters 1]);
parameters.batchnormSkip.Scale = initializeOnes([numFilters 1]);
state.batchnormSkip.TrainedMean = initializeZeros([numFilters 1]);
state.batchnormSkip.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the fully connected operation corresponding to the classification output, "fc1".
sz = [numClasses 6272];
numOut = numClasses;
numIn = 6272;
parameters.fc1.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc1.Bias = initializeZeros([numClasses 1]);

%% Initialize the parameters for the fully connected operation corresponding to the regression output, "fc2".
sz = [numResponses 6272];
numOut = numResponses;
numIn = 6272;
parameters.fc2.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc2.Bias = initializeZeros([numResponses 1]);

%% View the structure of the parameters.
parameters

%% View the parameters for the "conv1" operation.
parameters.conv1
%% View the structure of the state parameters.
state
%% step 3 Specify Training Options
%% Specify the training options. Train for 20 epochs with a mini-batch size of 128.
numEpochs = 20;
miniBatchSize = 128;

%% step 4 Train Model
% Use minibatchqueue to process and manage the mini-batches of images. For each mini-batch:
mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" "" ""]);

%% Initialize parameters for Adam.
trailingAvg = [];
trailingAvgSq = [];
%% Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

%% Train the model.
iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq)
    
    % Loop over mini-batches
    while hasdata(mbq)
    
        iteration = iteration + 1;
        
        [X,T1,T2] = next(mbq);
              
        % Evaluate the model loss, gradients, and state, using dlfeval and the
        % modelLoss function.
        [loss,gradients,state] = dlfeval(@modelloss,parameters,X,T1,T2,state);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,iteration);

        % Display the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(loss);
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end
%% step 5 Test Model 
% step 6 Model function
% step 7 Model Loss Function
% step 8 Mini-Batch Preprocessing Function