function [prob, correct, state] = MTCNN_model(parameters,X,doTraining,state)
% MTCNN Model as function 
% ARG:
% Input: 
%       Parameter
%       X 
%       doTraining flag
%       stage 
% Output:

% Initial operation 
%% Convolution - conv1

    weights = parameters.features_conv1_weight;
    bias = parameters.features_conv1_bias;

    Y = dlconv(X,weights,bias);

    % Batch normalization
    offset = parameters.batchnorm1.Offset;
    scale = parameters.batchnorm1.Scale;
    trainedMean = state.batchnorm1.TrainedMean;
    trainedVariance = state.batchnorm1.TrainedVariance;

    if doTraining
        [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
        % Update state
        state.batchnorm1.TrainedMean = trainedMean;
        state.batchnorm1.TrainedVariance = trainedVariance;
    else
        Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    end
    %    Y = maxpool(Y,[2 2],"Stride",2);
        Y = prelu(Y,parameters.features_prelu1.weight);
 %% Convolution - conv2
    weights = parameters.features_conv2_weight;
    bias = parameters.features_conv2_bias;

    Y = dlconv(Y,weights,bias);

    % Batch Normalize
    offset = parameters.batchnorm2.Offset;
    scale =  parameters.batchnorm2.Scale;
    trainedMean = state.batchnorm2.TrainedMean;
    trainedVariance = state.batchnorm2.TrainedVariance;

    if doTraining
        [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
        % Update state
        state.batchnorm2.TrainedMean = trainedMean;
        state.batchnorm2.TrainedVariance = trainedVariance;
    else
        Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    end

    Y = prelu(Y,parameters.features_prelu2.weight);

    %% Convolution - conv3
    weights = parameters.features_conv3_weight;
    bias = parameters.features_conv3_bias;

    Y = dlconv(Y,weights,bias);

    % Batch Normalize
    offset = parameters.batchnorm3.Offset;
    scale = parameters.batchnorm3.Scale;
    trainedMean = state.batchnorm3.TrainedMean;
    trainedVariance = state.batchnorm3.TrainedVariance;

    if doTraining
        [Y,trainedMean,trainedVariance] = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
        % Update state
        state.batchnorm3.TrainedMean = trainedMean;
        state.batchnorm3.TrainedVariance = trainedVariance;
    else
        Y = batchnorm(Y,offset,scale,trainedMean,trainedVariance);
    end

    Y = prelu(Y,parameters.features_prelu3.weight);
    %% Convolution - conv4_1
    weights = parameters.conv4_1_weight;
    bias = parameters.conv4_1_bias;

    Yprob = dlconv(Y,weights,bias,Padding="same");
%     % Batch Normalize
%     offset = parameters.batchnorm4_1.Offset;
%     scale = parameters.batchnorm4_1.Scale;
%     trainedMean = state.batchnorm4_1.TrainedMean;
%     trainedVariance = state.batchnorm4_1.TrainedVariance;
% 
%     if doTraining
%         [Yprob,trainedMean,trainedVariance] = batchnorm(Yprob,offset,scale,trainedMean,trainedVariance);
%         % Update state
%         state.batchnorm4_1.TrainedMean = trainedMean;
%         state.batchnorm4_1.TrainedVariance = trainedVariance;
%     else
%         Yprob = batchnorm(Yprob,offset,scale,trainedMean,trainedVariance);
%     end

    prob = softmax(Yprob);

      %% Convolution - conv4_2
    weights4_2 = parameters.conv4_2_weight;
    bias4_2 = parameters.conv4_2_bias;

    Yweight = dlconv(Y,weights4_2,bias4_2,Padding="same");

    % Batch Normalize
    offset = parameters.batchnorm4_2.Offset;
    scale = parameters.batchnorm4_2.Scale;
    trainedMean = state.batchnorm4_2.TrainedMean;
    trainedVariance = state.batchnorm4_2.TrainedVariance;

    if doTraining
        [Yweight,trainedMean,trainedVariance] = batchnorm(Yweight,offset,scale,trainedMean,trainedVariance);
        % Update state
        state.batchnorm4_2.TrainedMean = trainedMean;
        state.batchnorm4_2.TrainedVariance = trainedVariance;
    else
        Yweight = batchnorm(Yweight,offset,scale,trainedMean,trainedVariance);
    end

    
   
    correct = Yweight;
   

end