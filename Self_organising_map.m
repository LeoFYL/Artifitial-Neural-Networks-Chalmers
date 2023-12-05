% Reset environment
clc;
clearvars;

% Load dataset
irisFeatures = csvread('iris-data.csv');
irisLabels = csvread('iris-labels.csv');

% Initialize hyperparameters and network architecture
netConfig = struct('inputDims', [40, 40, 4], 'outputDims', [40, 40]);
learningRate = struct('initial', 0.1, 'decay', 0.01);
neighborhoodWidth = struct('initial', 10, 'decay', 0.05);

% Preprocess data
irisFeatures = normalizeData(irisFeatures);

% Training settings
numIterations = 10;
numDataPoints = numel(irisLabels);

% Initialize synaptic weights
synapticWeights = rand(netConfig.inputDims);
distanceMatrix = zeros(netConfig.outputDims);

% Training the network
for iteration = 1:numIterations
    currentLearningRate = learningRate.initial * exp(-learningRate.decay * iteration);
    currentSigma = neighborhoodWidth.initial * exp(-neighborhoodWidth.decay * iteration);
    
    for sampleIdx = 1:numDataPoints
        randomIndex = randi(numDataPoints);
        randomSample = irisFeatures(randomIndex,:);
        
        [minRow, minCol] = findNearestNeuron(synapticWeights, randomSample, distanceMatrix);
        
        % Adjust weights if within neighborhood
        if distanceMatrix(minRow, minCol) < 3 / currentSigma
            for row = 1:netConfig.inputDims(1)
                for col = 1:netConfig.inputDims(2)
                    influence = calculateInfluence([row, col], [minRow, minCol], currentSigma);
                    for depth = 1:netConfig.inputDims(3)
                        weightDelta = influence * (randomSample(depth) - synapticWeights(row, col, depth));
                        synapticWeights(row, col, depth) = synapticWeights(row, col, depth) + currentLearningRate * weightDelta;
                    end
                end
            end
        end
    end
end

% Generate predictions using the network
predictedLocationsInitial = zeros(numDataPoints, 2);
predictedLocationsTrained = zeros(numDataPoints, 2);

for idx = 1:numDataPoints
    [initialRow, initialCol] = findNearestNeuron(rand(netConfig.inputDims), irisFeatures(idx,:), distanceMatrix);
    [trainedRow, trainedCol] = findNearestNeuron(synapticWeights, irisFeatures(idx,:), distanceMatrix);
    
    predictedLocationsInitial(idx,:) = [initialRow, initialCol];
    predictedLocationsTrained(idx,:) = [trainedRow, trainedCol];
end

% Visualize results
clc;
plotResults(predictedLocationsInitial, predictedLocationsTrained, irisLabels);

%% Helper functions

function normalizedData = normalizeData(data)
    normalizedData = data ./ max(data);
end

function [nearestRow, nearestCol] = findNearestNeuron(weights, inputVector, distMat)
    for r = 1:size(weights, 1)
        for c = 1:size(weights, 2)
            % The squeeze function is used to reduce the third dimension of the weight matrix to a vector.
            distMat(r, c) = euclideanDist(squeeze(weights(r,c,:)), inputVector);
        end
    end
    [nearestRow, nearestCol] = find(distMat == min(distMat(:)), 1);
end


function distance = euclideanDist(vec1, vec2)
    % Make sure vec1 and vec2 are both column vectors.
    vec1 = vec1(:);
    vec2 = vec2(:);
    % Calculate Euclidean distance
    distance = sqrt(sum((vec1 - vec2) .^ 2));
end


function infValue = calculateInfluence(pos, bestMatchingUnitPos, sigma)
    infValue = exp(-norm(pos - bestMatchingUnitPos)^2 / (2 * sigma^2));
end

function plotResults(initialLocs, trainedLocs, labels)
    subplot(1,2,1);
    gscatter(initialLocs(:,1), initialLocs(:,2), labels, 'rgb', 'osd');
    title('Initial Weights');
    
    subplot(1,2,2);
    gscatter(trainedLocs(:,1), trainedLocs(:,2), labels, 'rgb', 'osd');
    title('Trained Weights');
    
    suptitle('Iris Data Classification');
    xlim([-5 45]);
    ylim([-5 45]);
end
