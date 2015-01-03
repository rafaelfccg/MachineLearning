function csvdata = parseCSV(csvfile)

%% to take a look at the data

% csvfile = 'data/seasons.csv';

csvdata = readtable(csvfile);
csvdata = csvdata{:,:};

end