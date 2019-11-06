clc
clear

%% convert original image to mat
% https://github.com/mks0601/V2V-PoseNet_RELEASE/blob/master/data/NYU/PNG2BIN.m

%train
dataset_dir = '.\nyu_hand_dataset_v2\train\';
save_dir = '.\nyu_hand_dataset_v2\Preprossed\train_nyu\';
tot_frame_num = 72757;

%test
% dataset_dir = '.\nyu_hand_dataset_v2\test\';
% save_dir = '.\nyu_hand_dataset_v2\Preprossed\test_nyu\';
% tot_frame_num = 8252;

kinect_index = 1;

for image_index = 1:tot_frame_num
    filename_prefix = sprintf('%d_%07d', kinect_index, image_index);

    if exist([dataset_dir, 'depth_', filename_prefix, '.png'], 'file')
        
        % The top 8 bits of depth are packed into green and the lower 8 bits into blue.
        depth = imread([dataset_dir, 'depth_', filename_prefix, '.png']);
        depth = uint16(depth(:,:,3)) + bitsll(uint16(depth(:,:,2)), 8);
        
%         fp_save = fopen([save_dir, 'depth_', filename_prefix, '.bin'],'w');
%         fwrite(fp_save,permute(depth,[2,1,3]),'float');
%         fclose(fp_save);
%         save(fullfile(save_dir, strcat(num2str(image_index), '.mat')),'depth');

        %delete(strcat(folderpath,img_name));
    end

end

