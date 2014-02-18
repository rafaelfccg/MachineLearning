function data = parseCSV(csvfile)

%% to take a look at the data

% csvfile = 'data/seasons.csv';

data = readtable(csvfile);
data = data{:,:};

end