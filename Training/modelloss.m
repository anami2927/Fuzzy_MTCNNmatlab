function [loss,gradients,state] = modelloss(parameters,X,T1,T2,state)
th = 0.6;
doTraining = true;
N = X(:,:,1,:);
N = dlarray(N,'SSCB');

[Y1,Y2,state] = MTCNN_model(parameters,N,doTraining,state);
% 
% % Transform Y1 and Y2 to outputs 
    % Y1 = extractdata(gather(Y1));
%     Y2 = extractdata(gather(Y2));
% % Face probabilities
%     faces = max(Y1(:,:,1,:),Y1(:,:,2,:)); %hard threshold to detect faces.
  %   faces(1,:) = Y1(:,:,1,:);
%     faces(2,:) = Y1(:,:,2,:);
%     faces(3,:) = Y1(:,:,3,:);
% 
%   
%     %faces = reshape(faces, [1 size(T1,2)]);
  %   faces = dlarray(faces,"SB");
% % 
      count = numel(Y1(:,:,1,:));
      ind = Y1(:,:,1,:) > Y1(:,:,2,:);
      faces = zeros(count);
      faces(ind) = 0;
      faces(~ind) = 1;
      faces = dlarray(faces,"SB");

lossLabels = crossentropy(faces,T1);

% T2 = reshape(T2, [size(T1,2) 14]);
% T2 = T2';
% T2 = T2(1:4,:);
% 
% offsets = Y2(:,:,1:4,:);
% 
% offsets = reshape(offsets,[4 size(T1,2)]);
% offsets = dlarray(offsets,"SB");
lossOffset = mse(Y2,Y2);

loss = lossLabels + 0.5*lossOffset; %total loss
gradients = dlgradient(dlarray(lossLabels),parameters);

end