function downloadTrainedNetwork(url,destination)
% The downloadTrainedNetwork function downloads a pretrained network.
%
% Copyright 2021-2023 The MathWorks, Inc.

if ~exist(destination,"dir")
    mkdir(destination);
end

[~,name,filetype] = fileparts(url);
netFileFullPath = fullfile(destination,name+filetype);

% Check for the existence of the file and download the file if it does not
% exist
if ~exist(netFileFullPath,"file")
    disp("Downloading pretrained network.");
    disp("This can take several minutes to download...");
    websave(netFileFullPath,url);

    % If the file is a ZIP file, extract it
    if filetype == ".zip"
        unzip(netFileFullPath,destination)
    end
    disp("Done.");

end