function [Y1,Y2,state] = mymodel(parameters,X,doTraining,state)

% Initial operations
% Convolution - conv1
weights = parameters.conv1.Weights;
bias = parameters.conv1.Bias;
Y = dlconv(X,weights,bias,Padding="same",Stride=2);
%Y = maxpool(Y,2,"Stride",2);


% Batch normalization, ReLU - batchnorm1, relu1
offset = parameters.batchnorm1.Offset;
scale = parameters.batchnorm1.Scale;
trainedMean = state.batchnorm1.TrainedMean;
trainedVariance = state.batchnorm1.TrainedVariance;

if doTraining
    % Normalize data across all observations for each channel independently
    % To speed up training of the convolutional neural network and reduce the sensitivity to network initialization, 
    % use batch normalization between convolution and nonlinear operations such as relu.
    [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm1.TrainedMean = trainedMean;
    state.batchnorm1.TrainedVariance = trainedVariance;
else
    Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
end

Y = relu(Y);


% Main branch operations
% Convolution - conv2
weights = parameters.conv2.Weights;
bias = parameters.conv2.Bias;
YnoSkip = dlconv(Y,weights,bias,Padding="same",Stride=2);

% Batch normalization, ReLU - batchnorm2, relu2
offset = parameters.batchnorm2.Offset;
scale = parameters.batchnorm2.Scale;
trainedMean = state.batchnorm2.TrainedMean;
trainedVariance = state.batchnorm2.TrainedVariance;

if doTraining
    [YnoSkip,trainedMean,trainedVariance] = batchnorm(YnoSkip,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm2.TrainedMean = trainedMean;
    state.batchnorm2.TrainedVariance = trainedVariance;
else
    YnoSkip = batchnorm(YnoSkip,offset,scale,trainedMean,trainedVariance);
end

YnoSkip = relu(YnoSkip);

% Convolution - conv3
weights = parameters.conv3.Weights;
bias = parameters.conv3.Bias;
YnoSkip = dlconv(YnoSkip,weights,bias,Padding="same");

% Batch normalization - batchnorm3
offset = parameters.batchnorm3.Offset;
scale = parameters.batchnorm3.Scale;
trainedMean = state.batchnorm3.TrainedMean;
trainedVariance = state.batchnorm3.TrainedVariance;

if doTraining
    [YnoSkip,trainedMean,trainedVariance] = batchnorm(YnoSkip,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm3.TrainedMean = trainedMean;
    state.batchnorm3.TrainedVariance = trainedVariance;
else
    YnoSkip = batchnorm(YnoSkip,offset,scale,trainedMean,trainedVariance);
end


% Skip connection operations
% Convolution, batch normalization (Skip connection) - convSkip, batchnormSkip
weights = parameters.convSkip.Weights;
bias = parameters.convSkip.Bias;
YSkip = dlconv(Y,weights,bias,Stride=2);

offset = parameters.batchnormSkip.Offset;
scale = parameters.batchnormSkip.Scale;
trainedMean = state.batchnormSkip.TrainedMean;
trainedVariance = state.batchnormSkip.TrainedVariance;

if doTraining
    [YSkip,trainedMean,trainedVariance] = batchnorm(YSkip,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnormSkip.TrainedMean = trainedMean;
    state.batchnormSkip.TrainedVariance = trainedVariance;
else
    YSkip = batchnorm(YSkip,offset,scale,trainedMean,trainedVariance);
end


% Final operations
% Addition, ReLU - addition, relu4
Y = YSkip + YnoSkip;
Y = relu(Y);

% Fully connect, softmax (labels) - fc1, softmax
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
Y1 = fullyconnect(Y,weights,bias);
Y1 = softmax(Y1);

% Fully connect (angles) - fc2
weights = parameters.fc2.Weights;
bias = parameters.fc2.Bias;
Y2 = fullyconnect(Y,weights,bias);

end