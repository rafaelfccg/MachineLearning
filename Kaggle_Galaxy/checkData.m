
% used to visualize and analyze data

datafile = 'E:\Datasets\Kaggle_Galaxy\training_solutions_rev1.csv';

data = parseCSV(datafile);

colNames = data.Properties.VariableNames;

% acess data only
% data.GalaxyID(1,1) : specify column name