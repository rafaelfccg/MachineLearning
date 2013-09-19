% Spectral Hashing
% Y. Weiss, A. Torralba, R. Fergus. 
% Advances in Neural Information Processing Systems, 2008.
%
% CONVENTIONS:
%    data points are row vectors.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% all data vectors are row based

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Create toy data:
% some parameters
Ntraining = 3000; % number training samples
Ntest = 3000; % number test samples
averageNumberNeighbors = 50; % number of groundtruth neighbors on training set (average)
aspectratio = 0.5; % aspect ratio between the two axis
loopbits = [2 4 8 16 32]; % try different number of bits for coding

% uniform distribution with larger range
data_range = 100;
Xtraining = rand([Ntraining,2]) * data_range; 
Xtraining(:,2) = aspectratio * Xtraining(:,2);
Xtest = rand([Ntest,2]) * data_range; 
Xtest(:,2) = aspectratio * Xtest(:,2);

% define ground-truth neighbors (this is only used for the evaluation):
DtrueTraining = distMat(Xtraining);
DtrueTestTraining = distMat(Xtest,Xtraining); % size = [Ntest x Ntraining]
Dball = sort(DtrueTraining,2);
Dball = mean(Dball(:,averageNumberNeighbors));  % mean distance for neighbors in training set
WtrueTestTraining = DtrueTestTraining < Dball;  % neighbors for testing sampels in training set
TrainingNeighbors = DtrueTraining < Dball;

train_pairs = cell(size(TrainingNeighbors,1), 2);
for i=1:size(TrainingNeighbors,1)
    
    train_pairs{i,1} = find(TrainingNeighbors(i,:) == 1);
    train_pairs{i,2} = find(TrainingNeighbors(i,:) == 0);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare parameters

feat_dim = 2;
bit_num = 16;

OWHParams.nbits = bit_num;
OWHParams.prev_weights = ones(1, OWHParams.nbits) / OWHParams.nbits;
OWHParams.cur_weights = OWHParams.prev_weights;
OWHParams.lamda = 0.1;
OWHParams.eta = 0.05;

old_params = OWHParams;

% randomly generate functions

LSHCoder.funcs = randn(bit_num, feat_dim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) demo online weighted hashing

% generate base hash code for each sample
train_codes = compress2Base(Xtraining, LSHCoder, 'LSH');
test_codes = compress2Base(Xtest, LSHCoder, 'LSH');

% learn weights in online fasion

for i=1:3000
    
    % randomly pick a triplet (training sample, positive sample, negative sample)
    train_id = max(1, int32( rand(1) * size(train_codes, 1) ));
    pos_id = max(1, int32( rand(1) * size(train_pairs{train_id, 1}, 2) ));
    pos_id = int32( train_pairs{train_id, 1}(1, pos_id) );
    neg_id = max(1, int32( rand(1) * size(train_pairs{train_id, 2}, 2) ));
    neg_id = int32( train_pairs{train_id, 2}(1, neg_id) );
    
    triplet.query_code = train_codes(train_id, :);
    triplet.pos_code = train_codes(pos_id, :);
    triplet.neg_code = train_codes(neg_id, :);
    
    % update weight
    OWHParams = weightLearner(OWHParams, triplet);
    
    %disp(OWHParams.cur_weights);
    disp(['Finish ' num2str(i) 'th update.']);
    
end

%disp(old_params.cur_weights);

% evaluation
% check best neighbors with weighted hamming distance and ground truth
% neighbors

% compute lsh hamming distance
lsh_dist = weightedHam(train_codes, train_codes, ones(1, bit_num));
[lsh_sorted_dist, lsh_sorted_idx] = sort(lsh_dist, 2);

lsh_inter = intersect(lsh_sorted_idx(1, find(lsh_sorted_dist(i, :)==0)), train_pairs{1,1});

% compute weighted hamming distance
owh_dist = weightedHam(train_codes, train_codes, OWHParams.cur_weights);
[owh_sorted_dist, owh_sorted_idx] = sort(owh_dist, 2);

% compute precision-recall with ground truth
% precision
for i=1:size(owh_sorted_idx, 1)
    
    owh_inter = intersect(owh_sorted_idx(1, find(owh_sorted_dist(i, :)==0)), train_pairs{1,1});
    disp(size(inter,1) / size(train_pairs{i,1},1));
    
end

