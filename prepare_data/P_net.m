%% Create the network with multiple output
layers = [imageInputLayer([28 28 1],'Normalization','none','Name','input')
          convolution2dLayer(3,10,Padding="same",Name='conv1')
          maxPooling2dLayer([2 2],Stride=[2 2])
          convolution2dLayer(3,16,Padding="same",Name='conv2')
          reluLayer
          convolution2dLayer(3,32,Padding="same",Name='conv3')
          reluLayer("Name",'relu2')
          convolution2dLayer([1 1],2,Padding="same",Name='conv4')
          softmaxLayer
          ];
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph, convolution2dLayer([1 1],4,Padding="same",Name='conv6'));
lgraph = connectLayers(lgraph,'relu2','conv6');


figure
plot(lgraph)
dlnet = dlnetwork(lgraph);
%% Training Data
XTrain = rand(28,28,1,50);% Input data (50 images of size 28x28x1)
YTrain1 = randi(2,50,2); % Regression Output data for Output 1
YTrain2 = randi(10,50,4); % Regression Output data for Output 2
dsXTrain = arrayDatastore(XTrain,'IterationDimension',4);
dsYTrain1 = arrayDatastore(YTrain1);
dsYTrain2 = arrayDatastore(YTrain2);
dsTrain = combine(dsXTrain,dsYTrain1,dsYTrain2);
%% Train the Network
numEpochs = 3;
miniBatchSize = 128;
plots = "training-progress";
mbq = minibatchqueue(dsTrain,'MiniBatchSize',miniBatchSize,'MiniBatchFcn', @preprocessData,'MiniBatchFormat',{'SSCB','',''});
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration");ylabel("Loss");grid on
end
trailingAvg = [];
trailingAvgSq = [];
iteration = 0;
start = tic;
% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq)
    
    % Loop over mini-batches
    while hasdata(mbq)
        
        iteration = iteration + 1;
        
        [dlX,dlY1,dlY2] = next(mbq);
                       
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function.
        [gradients,state,loss] = dlfeval(@modelGradients, dlnet, dlX, dlY1, dlY2);
        dlnet.State = state;
        
        % Update the network parameters using the Adam optimizer.
        [dlnet,trailingAvg,trailingAvgSq] = adamupdate(dlnet,gradients,trailingAvg,trailingAvgSq,iteration);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
%% Necessary function to train the network
function [gradients,state,loss] = modelGradients(dlnet,dlX,T1,T2)
[dlY1,dlY2,state] = forward(dlnet,dlX,'Outputs',["conv4" "conv6"]);
lossT1 = mse(dlY1,T1);
lossT2 = mse(dlY2,T2);
loss = 0.1*lossT1 + 0.1*lossT2;
gradients = dlgradient(loss,dlnet.Learnables);
end
function [X,Y1,Y2] = preprocessData(XCell,Y1Cell,Y2Cell)
    X = cat(4,XCell{:});
    Y1 = cat(2,Y1Cell{:});
    Y2 = cat(2,Y2Cell{:}); 
    
end