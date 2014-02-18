

%

datafile = 'E:\Datasets\Kaggle_Galaxy\training_solutions_rev1.csv';

data = parseCSV(datafile);

% recover original unweighted probability for each task

% 1_1 -> 7
data.Class7_1 = data.Class7_1 ./ data.Class1_1;
data.Class7_2 = data.Class7_2 ./ data.Class1_1;
data.Class7_3 = data.Class7_3 ./ data.Class1_1;

% 1_2 -> 2
data.Class2_1 = data.Class2_1 ./ data.Class1_2;
data.Class2_2 = data.Class2_2 ./ data.Class1_2;

% 2_1 -> 9
data.Class9_1 = data.Class9_1 ./ data.Class2_1;
data.Class9_2 = data.Class9_2 ./ data.Class2_1;
data.Class9_3 = data.Class9_3 ./ data.Class2_1;

% 2_2 -> 3
data.Class3_1 = data.Class3_1 ./ data.Class2_2;
data.Class3_2 = data.Class3_2 ./ data.Class2_2;

% 3 -> 4


% 4_1 -> 10

% 4_2 -> 5

% 5 -> 6

% 6_1 -> 8

% 7 -> 6

