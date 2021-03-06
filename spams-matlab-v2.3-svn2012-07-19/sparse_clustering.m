
clear all

start_spams

%%% train dictionary and get codes  
% paths

dataset_dir_home = 'H:\AmazonProductData\';
dataset_dir_lab = 'F:\Datasets\MobileProductSearch\Whole_Dataset\AmazonProductSet\';

img_dir = [dataset_dir_lab 'DemoDataSet_resized\']; 

cate_list_file = [dataset_dir_lab  'test_categories2.txt'];
cate_obj_file = [dataset_dir_lab 'uobjects.txt'];

img_desc_dir_home = 'H:\MobileVisualSearch\bow_descriptors\';
img_desc_dir_lab = 'F:\Results\MobileVisualSearch\bow_descriptors\';

res_dir_home = 'H:\MobileVisualSearch\';
res_dir_lab = 'F:\Results\MobileVisualSearch\';

res_cluster_dir = [res_dir_lab 'bow_clusters\sparse\'];

% params for sparse coding
param.K=50;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=4; % number of threads
param.batchsize=400;
param.iter=1000;  % let us see what happens after 1000 iterations.

% structures
training_objs = cell(0,0);

% read image files
fid = fopen(cate_obj_file, 'r');
while ~feof(fid)
    name = fscanf(fid, '%s', 1);
    num = fscanf(fid, '%d', 1);
    cate.name = name;
    cate.imgs = cell(num,1);
    for j=1:num
        dim = fscanf(fid, '%d %d');
        cate.imgs{j,1} = fscanf(fid, '%s', 1);
    end
    training_objs = [training_objs; cate];
end
fclose(fid);

% process each category
for i=1:size(training_objs,1)
    
    cur_cate_data = training_objs{i,1};
    % read descriptor for all images
    img_num = size(cur_cate_data.imgs,1);
    data = zeros(img_num, 128*64);
    for j=1:img_num
        desc_file = [img_desc_dir_lab cur_cate_data.imgs{j,1} '.imgsurf'];
        % read image descriptor
        fid = fopen(desc_file, 'r');
        dim = fscanf(fid, '%d', 1);
        data(j, :) = fscanf(fid, '%f', inf);
        fclose(fid);
        
        disp(['read ' desc_file]);
    end
    
    data = data';
    % create results dir
    res_cur_cluster_dir = [res_cluster_dir cur_cate_data.name '\'];
    mkdir(res_cur_cluster_dir);

    % build dictionary
    fprintf('Start to learn dictionary.\n');
    tic
    D = mexTrainDL(data, param);
    t=toc;
    fprintf('time of computation for Dictionary Learning: %f\n',t);
    
    % create directory for each dictionary item
    group_dirs = cell(param.K,1);
    for j=1:param.K
        res_dir = [res_cur_cluster_dir num2str(j) '\'];
        mkdir(res_dir);
        group_dirs{j,1} = res_dir;
    end

    % split images into groups based on distance to dictionary items
    best_img_id = zeros(1, param.K);    % save closest image id for each dictionary item (cluster center)
    best_img_dist = ones(1, param.K) * inf;
    for j=1:size(data, 2)
        diff = repmat(data(:,j), 1, param.K) - D;
        diff = sqrt(sum(diff.^2,1));
        [min_dist, min_id] = min(diff);
        if min_dist < best_img_dist(min_id)
            best_img_dist(min_id) = min_dist;
            best_img_id(min_id) = j;
        end
        % copy it to dir
        copyfile([img_dir cur_cate_data.imgs{j,1}], group_dirs{min_id,1});
    end
    
    % copy representative image to directory and save descriptors
    dict_file = [res_cur_cluster_dir 'dict.mat'];
    save(dict_file, 'D');
    for j=1:param.K
        id = int32(j);
        if best_img_id(1,id)<= 0
            continue;
        end
        copyfile([img_dir cur_cate_data.imgs{best_img_id(1,id), 1}], [res_cur_cluster_dir num2str(id) '.jpg']);
    end
    
end







param.approx=0;
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% ImD=displayPatches(D);
% subplot(1,3,1);
% imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);
drawnow;