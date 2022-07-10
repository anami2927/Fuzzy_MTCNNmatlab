function out = train_MTCNN(XTrain, T1Train, T2Train)
    % Training MTCNN model
    %% Load Data
    % test

    % Datastore for in-memory data
    T2Train = round(T2Train,2);
    dsXTrain = arrayDatastore(XTrain,"IterationDimension",4); 
    dsT1Train = arrayDatastore(T1Train);
    dsT2Train = arrayDatastore(T2Train);
    dsTrain = combine(dsXTrain,dsT1Train,dsT2Train);


    
%     classNames = categories(T1Train);
%     numClasses = numel(classNames);
%     numResponses = size(T2Train,2);
%     numObservations = numel(T1Train);

    %% Define MTCNN Model
% Define the following network that predicts both labels and angles of rotation.
% 
% A convolution-ReLU-l2-regularizers block with 10 3-by-3 filters.
% Maxpooling pool size 2 by 2 and stride 2
%
% A convolution-ReLU-l2-regularizers block with 16 3-by-3 filters.
% A convolution-ReLU-l2-regularizers block with 32 3-by-3 filters.

% For the output, prediced face with a convolution-l2-regularizers with 2 1 by 1 filter.
% For the output, bounding box with a convolution-l2-regularizers with 4 1 by 1 filter.
% For the output, predicted landmark with a convolution-l2-regularizers with 10 1 by 1 filter.

%% Initialize the parameters for the first convolution operation, "conv1".

    filterSize = [3 3];
    numChannels = 1;
    numFilters = 10;
    
    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;

    
    parameters.features_conv1_weight = initializeGlorot(sz,numOut,numIn); % random initial weights 
    parameters.features_conv1_bias = initializeZeros([numFilters 1]); % initial bias with zeros
% Initialze the parameters for prelu1
    sz = [1 1 numFilters];
    numOut = prod([1 1]) * numFilters;
    numIn = prod([1 1]) * numFilters;
    parameters.features_prelu1.weight = initializeGlorot(sz, numOut,numIn);

% Initialize the parameters and state for the first batch normalization operation, "batchnorm1".
    parameters.batchnorm1.Offset = initializeZeros([numFilters 1]);
    parameters.batchnorm1.Scale = initializeOnes([numFilters 1]);

% Initialize the batch normalization trained mean and trained variance states using the zeros and ones functions, respectively.
    state.batchnorm1.TrainedMean = initializeZeros([numFilters 1]);
    state.batchnorm1.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the second convolution operation, "conv2".
    filterSize = [3 3];
    numChannels = 10;
    numFilters = 16;
    
    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;

    parameters.features_conv2_weight  = initializeGlorot(sz,numOut,numIn);
    parameters.features_conv2_bias = initializeZeros([numFilters 1]);

% Initialze the parameters for prelu2
    sz = [1 1 numFilters];
    numOut = prod([1 1]) * numFilters;
    numIn = prod([1 1]) * numFilters;
    parameters.features_prelu2.weight = initializeGlorot(sz, numOut,numIn);

% Initialize the parameters and state for the second batch normalization operation, "batchnorm2".
  
    parameters.batchnorm2.Offset = initializeZeros([numFilters 1]);
    parameters.batchnorm2.Scale = initializeOnes([numFilters 1]);
    state.batchnorm2.TrainedMean = initializeZeros([numFilters 1]);
    state.batchnorm2.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the third convolution operation, "conv3".
   
    filterSize = [3 3];
    numChannels = 16;
    numFilters = 32;
    
    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;
    
    parameters.features_conv3_weight = initializeGlorot(sz,numOut,numIn);
    parameters.features_conv3_bias = initializeZeros([numFilters 1]);

% Initialze the parameters for prelu3
    sz = [1 1 numFilters];
    numOut = prod([1 1]) * numFilters;
    numIn = prod([1 1]) * numFilters;
    parameters.features_prelu3.weight = initializeGlorot(sz, numOut,numIn);

% Initialize the parameters and state for the third batch normalization operation, "batchnorm3".
    parameters.batchnorm3.Offset = initializeZeros([numFilters 1]);
    parameters.batchnorm3.Scale = initializeOnes([numFilters 1]);
    state.batchnorm3.TrainedMean = initializeZeros([numFilters 1]);
    state.batchnorm3.TrainedVariance = initializeOnes([numFilters 1]);

%% Initialize the parameters for the third convolution operation, "conv4_1".
    filterSize = [1 1];
    numChannels = 32;
    numFilters = 2;
    
    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;
    
    parameters.conv4_1_weight = initializeGlorot(sz,numOut,numIn);
    parameters.conv4_1_bias = initializeZeros([numFilters 1]);

    % Initialize the parameters and state for the third batch normalization operation, "batchnorm4_1".
    parameters.batchnorm4_1.Offset = initializeZeros([numFilters 1]);
    parameters.batchnorm4_1.Scale = initializeOnes([numFilters 1]);
    state.batchnorm4_1.TrainedMean = initializeZeros([numFilters 1]);
    state.batchnorm4_1.TrainedVariance = initializeOnes([numFilters 1]);

    %% Initialize the parameters for the third convolution operation, "conv4_2".
    filterSize = [1 1];
    numChannels = 32;
    numFilters = 4;
    
    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;
    
    parameters.conv4_2_weight = initializeGlorot(sz,numOut,numIn);
    parameters.conv4_2_bias = initializeZeros([numFilters 1]);

    %% Initialize the parameters and state for the third batch normalization operation, "batchnorm4_2".
    parameters.batchnorm4_2.Offset = initializeZeros([numFilters 1]);
    parameters.batchnorm4_2.Scale = initializeOnes([numFilters 1]);
    
    state.batchnorm4_2.TrainedMean = initializeZeros([numFilters 1]);
    state.batchnorm4_2.TrainedVariance = initializeOnes([numFilters 1]);

%% Training options: Specify the training options. Train for 20 epochs with a mini-batch size of 128.
    numEpochs = 30;
    miniBatchSize = 1;

    %% step 4 Train Model
% Use minibatchqueue to process and manage the mini-batches of images. For each mini-batch:
    mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    OutputEnvironment='gpu',...,
    OutputAsDlarray=1,...
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
learnRate = 0.001;
gradDecay = 0.6;
sqGradDecay = 0.95;
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
            trailingAvg,trailingAvgSq,iteration,learnRate,gradDecay,sqGradDecay);

        % Display the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(loss);
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end
end