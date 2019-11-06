%% set parameters
folderpath = '.\train_icvl\Depth\';
filepath = '.\train_icvl\labels.txt';
frameNum = 331006;

% folderpath = '.\test_icvl\Depth\';
% filepath = '.\test_icvl\icvl_test_list.txt';
% frameNum = 702+894;

%folderpath = '/home/gyeongsikmoon/workspace/Data/Hand_pose_estimation/ICVL/Testing/Depth/';
%filepath = '/home/gyeongsikmoon/workspace/Data/Hand_pose_estimation/ICVL/Testing/test_seq_2.txt';
%frameNum = 894;

save_dir = '.\train_icvl';

fp = fopen(filepath);
fid = 1;

tline = fgetl(fp);
while fid <= frameNum
    
    splitted = strsplit(tline);
    img_name = splitted{1};
    
    if exist(strcat(folderpath,img_name), 'file')
        img = imread(strcat(folderpath,img_name));
       
        fp_save = fopen(strcat(folderpath,img_name(1:size(img_name,2)-3),'bin'),'w');
        fwrite(fp_save,permute(img,[2,1,3]),'float');
        fclose(fp_save);
        
        save(fullfile(save_dir, strcat(num2str(fid), '.mat')),'img');
        
        %delete(strcat(folderpath,img_name));
    end

    tline = fgetl(fp);
    fid = fid + 1;
end

fclose(fp);







